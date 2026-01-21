#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import onnxruntime as ort
import torch

from utils import (
    build_source_ids,
    compute_feat,
    load_and_resample_audio,
    pick_providers,
    sample_token,
    select_device,
    setup_tokenizer,
    EncoderAdaptorOnnxModel,
    EmbeddingOnnxIOB,
    UnifiedKvDeltaLLMOnnxIOB,
    device_from_str,
)


@dataclass
class UtterancePrep:
    wav: str
    audio_dur: float
    source_ids: np.ndarray
    fbank_beg: int
    fake_len: int
    enc_out: np.ndarray   # [1,audio_len,D] numpy


@dataclass
class StreamState:
    wav: str
    audio_dur: float
    generated: List[int]
    next_logits: np.ndarray  # [vocab] on CPU
    done: bool = False


class BatchGroup:
    def __init__(
        self,
        prompt_len: int,
        device: torch.device,
        llm: UnifiedKvDeltaLLMOnnxIOB,
        eos_token_id: Optional[int],
        im_end_token_id: Optional[int],
    ):
        self.prompt_len = int(prompt_len)
        self.device = device
        self.llm = llm
        self.eos = eos_token_id
        self.im_end = im_end_token_id

        self.streams: List[StreamState] = []
        self.caches_k: List[torch.Tensor] = []
        self.caches_v: List[torch.Tensor] = []

        self.past_len = int(prompt_len)
        self.step = 0

        # reusable outputs for decode step (B may shrink after compaction; rebuild when needed)
        self._out_B = 0
        self._logits_out: Optional[torch.Tensor] = None
        self._kd_out: Optional[List[torch.Tensor]] = None
        self._vd_out: Optional[List[torch.Tensor]] = None

    def active_count(self) -> int:
        return sum(0 if s.done else 1 for s in self.streams)

    def ensure_outputs(self, batch: int, seq: int):
        if self._out_B == int(batch) and self._logits_out is not None:
            # seq for decode is always 1; keep simple
            if self._kd_out is not None and self._vd_out is not None and int(self._kd_out[0].shape[1]) == int(seq):
                return
        self._out_B = int(batch)
        self._logits_out, self._kd_out, self._vd_out = self.llm.alloc_outputs(batch=batch, seq=seq, device=self.device)

    def compact(self):
        keep = [i for i, s in enumerate(self.streams) if not s.done]
        if len(keep) == 0:
            self.streams = []
            self.caches_k = []
            self.caches_v = []
            self._out_B = 0
            self._logits_out = None
            self._kd_out = None
            self._vd_out = None
            return
        if len(keep) == len(self.streams):
            return
        idx = torch.tensor(keep, device=self.device, dtype=torch.int64)
        self.streams = [self.streams[i] for i in keep]
        self.caches_k = [ck.index_select(0, idx) for ck in self.caches_k]
        self.caches_v = [cv.index_select(0, idx) for cv in self.caches_v]
        # outputs must be rebuilt for new B
        self._out_B = 0
        self._logits_out = None
        self._kd_out = None
        self._vd_out = None


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--encoder-adaptor-model", type=str, required=True)
    p.add_argument("--embedding-model", type=str, required=True)
    p.add_argument("--llm-model", type=str, required=True)
    p.add_argument("--llm-tokenizer", type=str, required=True)

    p.add_argument("--wave", type=str, nargs="+", required=True)
    p.add_argument("--prompt", type=str, default="语音转写：")

    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--encoder-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--embedding-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--llm-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")

    return p.parse_args()


def main():
    args = get_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    enc_dev = select_device(args.encoder_device)
    llm_dev = select_device(args.llm_device, model_path=args.llm_model)
    emb_dev = select_device(args.embedding_device)
    if emb_dev == "auto":
        emb_dev = llm_dev  # keep same as llm to avoid transfers

    device = device_from_str("cuda" if llm_dev == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    tokenizer, eos_token_id, im_end_token_id = setup_tokenizer(args.llm_tokenizer)

    encoder = EncoderAdaptorOnnxModel(args.encoder_adaptor_model, device=enc_dev)
    embedding = EmbeddingOnnxIOB(args.embedding_model, device=emb_dev)
    llm = UnifiedKvDeltaLLMOnnxIOB(args.llm_model, device=llm_dev)

    print(f"[DEV] encoder={enc_dev}, embedding={emb_dev}, llm={llm_dev}, torch_device={device.type}")
    print(f"[LLM] quant={llm.quant_type}, layers={llm.num_layers}, max_total_len={llm.max_total_len}, kv_heads={llm.num_kv_heads}, head_dim={llm.head_dim}, vocab={llm.vocab_size}")

    # preprocess per utterance (encoder + ids)  
    preps: List[UtterancePrep] = []
    t0 = time.time()
    for wav in args.wave:
        samples, sr = load_and_resample_audio(wav)

        audio_dur = float(len(samples) / sr)
        feats = compute_feat(samples, sr, encoder.window_size, encoder.window_shift)[None, ...]
        enc_out = encoder(feats)  # numpy [1,T,D]
        enc_out = np.where(np.isfinite(enc_out), enc_out, 0.0)

        system_prompt = "You are a helpful assistant."
        user_prompt = f"{args.prompt}<|startofspeech|>!!<|endofspeech|>"
        audio_len = int(enc_out.shape[1])

        source_ids, fbank_beg, fake_len = build_source_ids(tokenizer, system_prompt, user_prompt, audio_len)
        preps.append(UtterancePrep(wav=wav, audio_dur=audio_dur, source_ids=source_ids, fbank_beg=fbank_beg, fake_len=fake_len, enc_out=enc_out))

    # Group by prompt_len (cache_position has no batch dim, so prefill can only batch same prompt_len)
    by_len: Dict[int, List[UtterancePrep]] = {}
    for p in preps:
        by_len.setdefault(int(p.source_ids.shape[0]), []).append(p)

    groups: List[BatchGroup] = []

    # prefill per group (batch)  
    for prompt_len, items in sorted(by_len.items(), key=lambda x: x[0]):
        B = len(items)
        S = int(prompt_len)

        if llm.max_total_len > 0 and S >= llm.max_total_len:
            raise RuntimeError(f"prompt_len={S} >= max_total_len={llm.max_total_len}. wavs={[it.wav for it in items]}")

        input_ids_np = np.stack([it.source_ids for it in items], axis=0).astype(np.int64, copy=False)  # [B,S]
        input_ids = torch.from_numpy(input_ids_np).to(device=device, dtype=torch.int64).contiguous()

        text_embeds = embedding.forward_ids(input_ids).to(dtype=llm.input_torch_dtype)
        text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=0.0, neginf=0.0)

        # fill audio span
        enc_stack = np.concatenate([it.enc_out for it in items], axis=0)  # [B,T,D]
        enc_t = torch.from_numpy(enc_stack).to(device=device, dtype=llm.input_torch_dtype).contiguous()

        for b, it in enumerate(items):
            fl = int(it.fake_len)
            if fl > enc_t.shape[1]:
                fl = int(enc_t.shape[1])
            text_embeds[b, it.fbank_beg:it.fbank_beg + fl, :] = enc_t[b, :fl, :]

        inputs_embeds = text_embeds.contiguous()

        caches_k, caches_v = llm.alloc_caches(batch=B, device=device)
        cache_position = torch.arange(0, S, device=device, dtype=torch.int64)  # shared for batch
        attention_mask = torch.ones((B, S), device=device, dtype=torch.int64)

        # pre-alloc outputs (prefill seq=S, logits is [B,1,V])
        logits_out, kd_out, vd_out = llm.alloc_outputs(batch=B, seq=S, device=device)

        logits, k_deltas, v_deltas = llm.run_iobinding(
            inputs_embeds, attention_mask, cache_position,
            caches_k, caches_v,
            logits_out, kd_out, vd_out
        )

        for i in range(llm.num_layers):
            caches_k[i][:, 0:S, :, :].copy_(k_deltas[i])
            caches_v[i][:, 0:S, :, :].copy_(v_deltas[i])

        g = BatchGroup(prompt_len=S, device=device, llm=llm, eos_token_id=eos_token_id, im_end_token_id=im_end_token_id)
        g.caches_k = caches_k
        g.caches_v = caches_v

        # logits-last: [B,1,V] -> take [:,0,:]
        logits_last = logits[:, 0, :].detach().float().cpu().numpy()  # [B,V]
        for b, it in enumerate(items):
            st = StreamState(wav=it.wav, audio_dur=it.audio_dur, generated=[], next_logits=logits_last[b], done=False)
            g.streams.append(st)

        groups.append(g)

    # micro-batch decode (per group lockstep)  
    for g in groups:
        if g.active_count() == 0:
            continue
        g.step = 0

    final_results: Dict[str, List[int]] = {}

    total_audio = sum(p.audio_dur for p in preps)
    decode_start = time.time()

    for _global in range(args.max_new_tokens):
        any_active = False

        for g in groups:
            if g.active_count() == 0:
                continue
            if llm.max_total_len > 0 and g.past_len >= llm.max_total_len:
                for s in g.streams:
                    s.done = True
                continue

            any_active = True

            # sample tokens for active streams
            toks: List[int] = []
            for s in g.streams:
                if s.done:
                    continue
                tok = sample_token(
                    s.next_logits,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=g.eos,
                    im_end_token_id=g.im_end,
                    step=g.step,
                )
                s.generated.append(tok)

                if g.step > 0 and ((g.eos is not None and tok == g.eos) or (g.im_end is not None and tok == g.im_end)):
                    s.done = True
                    final_results[s.wav] = list(s.generated)
                    continue

                toks.append(tok)

            # remove finished ones (compaction) before running ORT
            if g.active_count() == 0:
                g.compact()
                continue
            if any(s.done for s in g.streams):
                g.compact()
                if g.active_count() == 0:
                    continue
                toks = [s.generated[-1] for s in g.streams if not s.done]

            B = len(g.streams)
            input_ids = torch.tensor(np.array(toks, dtype=np.int64).reshape(B, 1), device=g.device, dtype=torch.int64).contiguous()
            tok_embeds = embedding.forward_ids(input_ids).to(dtype=llm.input_torch_dtype).contiguous()

            cache_position = torch.tensor([g.past_len], device=g.device, dtype=torch.int64)  # shared for batch
            attention_mask = torch.ones((B, g.past_len + 1), device=g.device, dtype=torch.int64)

            # reuse outputs for decode (seq=1)
            g.ensure_outputs(batch=B, seq=1)
            logits_out = g._logits_out
            kd_out = g._kd_out
            vd_out = g._vd_out
            assert logits_out is not None and kd_out is not None and vd_out is not None

            logits_step, k_deltas, v_deltas = llm.run_iobinding(
                tok_embeds, attention_mask, cache_position,
                g.caches_k, g.caches_v,
                logits_out, kd_out, vd_out
            )

            for i in range(llm.num_layers):
                g.caches_k[i][:, g.past_len:g.past_len + 1, :, :].copy_(k_deltas[i])
                g.caches_v[i][:, g.past_len:g.past_len + 1, :, :].copy_(v_deltas[i])

            # logits-last: [B,1,V] -> take [:,0,:]
            next_logits = logits_step[:, 0, :].detach().float().cpu().numpy()
            for b, s in enumerate(g.streams):
                s.next_logits = next_logits[b]

            g.past_len += 1
            g.step += 1

        if not any_active:
            break

    for g in groups:
        for s in g.streams:
            if s.wav not in final_results:
                final_results[s.wav] = list(s.generated)

    decode_end = time.time()
    total_time = decode_end - t0
    decode_time = decode_end - decode_start
    rtf = (total_time / total_audio) if total_audio > 0 else 0.0

    print("\n===== RESULTS =====")
    for wav in args.wave:
        toks = final_results.get(wav, [])
        txt = ""
        if toks:
            txt = tokenizer.decode(toks, skip_special_tokens=True)
            txt = txt.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
            txt = " ".join(txt.split())
        print(f"\n--- {wav} ---")
        print(txt)

    print(f"\n[TIME] total_audio={total_audio:.3f}s, wall_total={total_time:.3f}s, wall_decode={decode_time:.3f}s, RTF={rtf:.3f}")


if __name__ == "__main__":
    main()
