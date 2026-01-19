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
)

def _np_dtype_from_ort(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    if "int64" in s:
        return np.int64
    raise RuntimeError(f"Unsupported ORT type: {ort_type}")


def _torch_dtype_from_np(np_dtype: np.dtype):
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.int64:
        return torch.int64
    raise RuntimeError(f"Unsupported numpy dtype: {np_dtype}")


def _device_from_str(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


def _bind_torch_input(io_binding, name: str, t: torch.Tensor):
    t = t.contiguous()
    dev = t.device
    device_type = "cuda" if dev.type == "cuda" else "cpu"
    device_id = int(dev.index or 0) if dev.type == "cuda" else 0
    np_dtype = (
        np.float16 if t.dtype == torch.float16 else
        np.float32 if t.dtype == torch.float32 else
        np.int64 if t.dtype == torch.int64 else None
    )
    if np_dtype is None:
        raise RuntimeError(f"Unsupported torch dtype for bind_input: {t.dtype}")
    io_binding.bind_input(
        name=name,
        device_type=device_type,
        device_id=device_id,
        element_type=np_dtype,
        shape=tuple(t.shape),
        buffer_ptr=t.data_ptr(),
    )


def _bind_torch_output(io_binding, name: str, t: torch.Tensor):
    t = t.contiguous()
    dev = t.device
    device_type = "cuda" if dev.type == "cuda" else "cpu"
    device_id = int(dev.index or 0) if dev.type == "cuda" else 0
    np_dtype = (
        np.float16 if t.dtype == torch.float16 else
        np.float32 if t.dtype == torch.float32 else
        np.int64 if t.dtype == torch.int64 else None
    )
    if np_dtype is None:
        raise RuntimeError(f"Unsupported torch dtype for bind_output: {t.dtype}")
    io_binding.bind_output(
        name=name,
        device_type=device_type,
        device_id=device_id,
        element_type=np_dtype,
        shape=tuple(t.shape),
        buffer_ptr=t.data_ptr(),
    )

class EncoderAdaptorOnnxModel:
    def __init__(self, filename: str, device: str = "auto"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.window_size = int(meta.get("lfr_window_size", 7))
        self.window_shift = int(meta.get("lfr_window_shift", 6))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out_name], {self.in_name: x})[0]


class EmbeddingOnnxIOB:
    def __init__(self, filename: str, device: str = "auto"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = {o.name: o for o in self.sess.get_outputs()}
        self.in_np_dtype = _np_dtype_from_ort(ins[self.in_name].type)          # int64
        self.out_np_dtype = _np_dtype_from_ort(outs[self.out_name].type)       # fp16/fp32
        self.out_torch_dtype = _torch_dtype_from_np(self.out_np_dtype)
        out_shape = self.sess.get_outputs()[0].shape
        self.embed_dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else None
        if self.embed_dim is None:
            raise RuntimeError("Embedding output dim is dynamic; please export with static embed dim.")

    def forward_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        out = torch.empty((B, S, self.embed_dim), device=input_ids.device, dtype=self.out_torch_dtype).contiguous()
        io = self.sess.io_binding()
        _bind_torch_input(io, self.in_name, input_ids)
        _bind_torch_output(io, self.out_name, out)
        self.sess.run_with_iobinding(io)
        return out


class UnifiedKvDeltaLLMOnnxIOB:
    def __init__(self, filename: str, device: str = "auto"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        meta = self.sess.get_modelmeta().custom_metadata_map

        self.quant_type = str(meta.get("quantization_type", ""))
        self.num_layers = int(meta.get("num_layers", 0) or 0)
        self.max_total_len = int(meta.get("max_total_len", 0) or 0)
        self.num_kv_heads = int(meta.get("num_kv_heads", 0) or 0)
        self.head_dim = int(meta.get("head_dim", 0) or 0)
        self.vocab_size = int(meta.get("vocab_size", 0) or 0)
        self.logits_mode = str(meta.get("logits_mode", ""))  # expect "last"

        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = {o.name: o for o in self.sess.get_outputs()}
        self.input_np_dtype = _np_dtype_from_ort(ins["inputs_embeds"].type)
        self.cache_np_dtype = _np_dtype_from_ort(ins["cache_key_0"].type)
        self.logits_np_dtype = _np_dtype_from_ort(outs["logits"].type)

        self.input_torch_dtype = _torch_dtype_from_np(self.input_np_dtype)
        self.cache_torch_dtype = _torch_dtype_from_np(self.cache_np_dtype)
        self.logits_torch_dtype = _torch_dtype_from_np(self.logits_np_dtype)

        if self.num_layers <= 0:
            self.num_layers = len([k for k in ins.keys() if k.startswith("cache_key_")])

        if self.vocab_size <= 0:
            lshape = outs["logits"].shape
            if isinstance(lshape[-1], int):
                self.vocab_size = int(lshape[-1])

        if self.max_total_len <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise RuntimeError("Missing max_total_len/num_kv_heads/head_dim in metadata.")
        if self.vocab_size <= 0:
            raise RuntimeError("Missing vocab_size in metadata.")

        # logits-last expected: [B,1,V]
        # allow missing meta for old models, but enforce runtime shapes in run_iobinding
        if self.logits_mode and self.logits_mode != "last":
            print(f"[WARN] logits_mode meta is '{self.logits_mode}', but this runner assumes last-token logits.")

    def alloc_caches(self, batch: int, device: torch.device):
        caches_k, caches_v = [], []
        for _ in range(self.num_layers):
            ck = torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim),
                             device=device, dtype=self.cache_torch_dtype).contiguous()
            cv = torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim),
                             device=device, dtype=self.cache_torch_dtype).contiguous()
            caches_k.append(ck)
            caches_v.append(cv)
        return caches_k, caches_v

    # Allocate reusable IOB outputs for a fixed (B,S) call.
    def alloc_outputs(self, batch: int, seq: int, device: torch.device):
        # logits-last: [B,1,V] regardless of seq
        logits = torch.empty((batch, 1, self.vocab_size), device=device, dtype=self.logits_torch_dtype).contiguous()
        key_deltas, val_deltas = [], []
        for _ in range(self.num_layers):
            kd = torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous()
            vd = torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous()
            key_deltas.append(kd)
            val_deltas.append(vd)
        return logits, key_deltas, val_deltas

    def run_iobinding(
        self,
        inputs_embeds: torch.Tensor,       # [B,S,D]
        attention_mask: torch.Tensor,      # [B,total]
        cache_position: torch.Tensor,      # [S]   (no batch dim)
        caches_k: List[torch.Tensor],      # each [B,max_total,kv,hd]
        caches_v: List[torch.Tensor],
        logits_out: torch.Tensor,          # [B,1,V]
        key_deltas_out: List[torch.Tensor],# each [B,S,kv,hd]
        val_deltas_out: List[torch.Tensor],
    ):
        B = int(inputs_embeds.shape[0])
        S = int(inputs_embeds.shape[1])

        # shape guards (catch mismatched pre-alloc early)
        if int(logits_out.shape[0]) != B or int(logits_out.shape[1]) != 1 or int(logits_out.shape[2]) != self.vocab_size:
            raise RuntimeError(f"logits_out shape mismatch: expect ({B},1,{self.vocab_size}), got {tuple(logits_out.shape)}")
        for i in range(self.num_layers):
            if tuple(key_deltas_out[i].shape) != (B, S, self.num_kv_heads, self.head_dim):
                raise RuntimeError(f"key_delta_out[{i}] shape mismatch: got {tuple(key_deltas_out[i].shape)}, expect ({B},{S},{self.num_kv_heads},{self.head_dim})")
            if tuple(val_deltas_out[i].shape) != (B, S, self.num_kv_heads, self.head_dim):
                raise RuntimeError(f"value_delta_out[{i}] shape mismatch: got {tuple(val_deltas_out[i].shape)}, expect ({B},{S},{self.num_kv_heads},{self.head_dim})")

        io = self.sess.io_binding()
        _bind_torch_input(io, "inputs_embeds", inputs_embeds)
        _bind_torch_input(io, "attention_mask", attention_mask)
        _bind_torch_input(io, "cache_position", cache_position)
        for i in range(self.num_layers):
            _bind_torch_input(io, f"cache_key_{i}", caches_k[i])
            _bind_torch_input(io, f"cache_value_{i}", caches_v[i])

        _bind_torch_output(io, "logits", logits_out)
        for i in range(self.num_layers):
            _bind_torch_output(io, f"key_delta_{i}", key_deltas_out[i])
            _bind_torch_output(io, f"value_delta_{i}", val_deltas_out[i])

        self.sess.run_with_iobinding(io)
        return logits_out, key_deltas_out, val_deltas_out


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

    device = _device_from_str("cuda" if llm_dev == "cuda" else "cpu")
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
