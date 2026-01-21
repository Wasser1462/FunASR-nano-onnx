#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import onnxruntime as ort
import torch

from voice_activity_detector import SileroVAD
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


def _run_vad_segments_1p1(
    vad_model: SileroVAD,
    samples: np.ndarray,
    sr: int,
    pad_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
    min_seg_sec: float = 0.0,
    max_seg_sec: float = 20.0,
) -> List[Tuple[int, int]]:
    if sr != 16000:
        raise ValueError(f"Expected sr=16000 for VAD. Given: {sr}")

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    
    max_val = np.abs(samples).max()
    if max_val > 1.0:
        samples = samples / max_val

    vad_model.reset()

    window_size_samples = vad_model.get_window_size()
    window_shift_samples = vad_model.get_window_shift()
    n_total = int(len(samples))
    
    max_seg_samples = int(round(float(max_seg_sec) * sr))
    
    original_threshold = vad_model.threshold
    original_min_silence_samples = vad_model.min_silence_samples
    original_min_silence_duration = vad_model.min_silence_duration
    
    new_threshold = 0.9
    new_min_silence_duration = 0.1
    new_min_silence_samples = int(round(new_min_silence_duration * sr))

    speech_segments = []
    speech_start = None
    buffer_tail = 0
    chunk_size = window_shift_samples * 10
    buffer_head = 0
    buffer_size = 0
    
    i = 0
    while i < n_total - window_size_samples + 1:
        remaining = n_total - window_size_samples - i
        windows_in_chunk = min(chunk_size // window_shift_samples, 
                              (remaining // window_shift_samples) + 1)
        
        if buffer_size > max_seg_samples:
            vad_model.threshold = new_threshold
            vad_model.min_silence_duration = new_min_silence_duration
            vad_model.min_silence_samples = new_min_silence_samples
        else:
            vad_model.threshold = original_threshold
            vad_model.min_silence_duration = original_min_silence_duration
            vad_model.min_silence_samples = original_min_silence_samples
        
        is_speech = False
        chunk_end = i
        
        for w in range(windows_in_chunk):
            window_start = i + w * window_shift_samples
            if window_start + window_size_samples > n_total:
                break
            
            chunk = samples[window_start:window_start + window_size_samples]
            this_window_is_speech = vad_model.is_speech(chunk)
            is_speech = is_speech or this_window_is_speech
            chunk_end = window_start + window_shift_samples
        
        buffer_tail = chunk_end
        buffer_size = buffer_tail - buffer_head
        
        if is_speech:
            if speech_start is None:
                min_speech_samples = vad_model.min_speech_samples
                lookback = 2 * window_size_samples + min_speech_samples
                speech_start = max(buffer_head, buffer_tail - lookback)
        else:
            if speech_start is not None:
                min_silence_samples = vad_model.min_silence_samples
                segment_end = buffer_tail - min_silence_samples
                
                if segment_end > speech_start:
                    speech_segments.append((speech_start, segment_end))
                
                buffer_head = segment_end
                buffer_size = buffer_tail - buffer_head
                speech_start = None
            
            if speech_start is None:
                end = buffer_tail - 2 * window_size_samples - vad_model.min_speech_samples
                samples_to_pop = max(0, end - buffer_head)
                if samples_to_pop > 0:
                    buffer_head += samples_to_pop
                    buffer_size = buffer_tail - buffer_head
        
        i = chunk_end

    if speech_start is not None:
        segment_end = min(buffer_tail, n_total)
        if segment_end > speech_start:
            speech_segments.append((speech_start, segment_end))

    if len(speech_segments) == 0:
        return []

    speech_segments.sort(key=lambda x: x[0])

    pad = int(round(float(pad_sec) * sr))
    if pad > 0:
        padded = []
        for s, e in speech_segments:
            ss = max(0, s - pad)
            ee = min(n_total, e + pad)
            padded.append((ss, ee))
        speech_segments = padded

    merge_gap = int(round(float(merge_gap_sec) * sr))
    if merge_gap > 0:
        merged = []
        cur_s, cur_e = speech_segments[0]
        for s, e in speech_segments[1:]:
            if s <= cur_e + merge_gap:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        speech_segments = merged

    min_seg = int(round(float(min_seg_sec) * sr))
    if min_seg > 0:
        speech_segments = [(s, e) for (s, e) in speech_segments if (e - s) >= min_seg]

    return speech_segments


def _split_segment_with_overlap(ss: int, ee: int, sr: int, max_len_sec: float, overlap_sec: float):
    max_len = int(round(max_len_sec * sr))
    overlap = int(round(overlap_sec * sr))
    if max_len <= 0:
        return [(ss, ee)]
    if overlap < 0:
        overlap = 0
    if overlap >= max_len:
        overlap = max_len // 2

    if ee - ss <= max_len:
        return [(ss, ee)]

    step = max_len - overlap
    out = []
    cur = ss
    while cur < ee:
        nxt = min(cur + max_len, ee)
        out.append((cur, nxt))
        if nxt >= ee:
            break
        cur = nxt - overlap
    return out


def _dedup_by_overlap(prev: str, cur: str, max_k: int = 100) -> str:
    prev = prev or ""
    cur = cur or ""
    if not prev or not cur:
        return cur
    
    prev_clean = re.sub(r'[\s，。！？、；：]', '', prev)
    cur_clean = re.sub(r'[\s，。！？、；：]', '', cur)
    
    if not prev_clean or not cur_clean:
        return cur
    
    kmax = min(len(prev_clean), len(cur_clean), max_k)
    for k in range(kmax, 3, -1):
        if prev_clean.endswith(cur_clean[:k]):
            non_space_count = 0
            for i, char in enumerate(cur):
                if char not in ' \t\n，。！？、；：':
                    non_space_count += 1
                    if non_space_count >= k:
                        remaining = cur[i+1:].lstrip()
                        if remaining and remaining[0] not in '，。！？、；：':
                            return remaining
                        return remaining
            return cur[min(k, len(cur)):]
    return cur


def _sample_tokens_fast(
    logits_2d: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if logits_2d.ndim != 2:
        raise ValueError(f"logits_2d must be [B,V], got {tuple(logits_2d.shape)}")

    if temperature is None:
        temperature = 0.0
    if top_p is None:
        top_p = 1.0

    temperature = float(temperature)
    top_p = float(top_p)

    if temperature <= 0.0:
        return torch.argmax(logits_2d, dim=-1).to(dtype=torch.int64)

    x = logits_2d / temperature

    if top_p >= 1.0:
        probs = torch.softmax(x, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=torch.int64)

    sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    keep = cumsum <= top_p
    keep[:, 0] = True

    masked_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    denom = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    masked_probs = masked_probs / denom

    sampled_in_sorted = torch.multinomial(masked_probs, num_samples=1).squeeze(1)
    sampled_token = sorted_idx.gather(1, sampled_in_sorted.view(-1, 1)).squeeze(1)
    return sampled_token.to(dtype=torch.int64)




@dataclass
class UtterancePrep:
    key: str
    parent_wav: str
    seg_start: float
    seg_end: float
    audio_dur: float
    source_ids: np.ndarray
    fbank_beg: int
    fake_len: int
    enc_out: np.ndarray


@dataclass
class StreamState:
    key: str
    parent_wav: str
    seg_start: float
    seg_end: float
    audio_dur: float
    generated: List[int]
    done: bool = False


class BatchGroup:
    def __init__(
        self,
        prompt_len: int,
        device: torch.device,
        llm: UnifiedKvDeltaLLMOnnxIOB,
        eos_token_id: Optional[int],
        im_end_token_id: Optional[int],
        compact: bool = False,
    ):
        self.prompt_len = int(prompt_len)
        self.device = device
        self.llm = llm
        self.eos = eos_token_id
        self.im_end = im_end_token_id
        self.compact_enabled = bool(compact)

        self.streams: List[StreamState] = []
        self.caches_k: List[torch.Tensor] = []
        self.caches_v: List[torch.Tensor] = []

        self.past_len = int(prompt_len)
        self.step = 0

        self.logits_out: Optional[torch.Tensor] = None
        self.kd_out: Optional[List[torch.Tensor]] = None
        self.vd_out: Optional[List[torch.Tensor]] = None
        self.cache_position_1 = torch.empty((1,), device=self.device, dtype=torch.int64)

    def active_count(self) -> int:
        return sum(0 if s.done else 1 for s in self.streams)

    def ensure_decode_buffers(self, batch: int, seq: int):
        B = int(batch)
        S = int(seq)

        if self.logits_out is None or tuple(self.logits_out.shape) != (B, 1, self.llm.vocab_size):
            self.logits_out = self.llm.alloc_logits(batch=B, device=self.device)

        if self.kd_out is None or self.vd_out is None:
            self.kd_out, self.vd_out = self.llm.alloc_kv_deltas(batch=B, seq=S, device=self.device)
            return

        # If shapes mismatch, rebuild
        if tuple(self.kd_out[0].shape) != (B, S, self.llm.num_kv_heads, self.llm.head_dim):
            self.kd_out, self.vd_out = self.llm.alloc_kv_deltas(batch=B, seq=S, device=self.device)

    def compact(self):
        keep = [i for i, s in enumerate(self.streams) if not s.done]
        if len(keep) == 0:
            self.streams = []
            self.caches_k = []
            self.caches_v = []
            self.logits_out = None
            self.kd_out = None
            self.vd_out = None
            return

        if len(keep) == len(self.streams):
            return

        idx = torch.tensor(keep, device=self.device, dtype=torch.int64)

        self.streams = [self.streams[i] for i in keep]
        self.caches_k = [ck.index_select(0, idx) for ck in self.caches_k]
        self.caches_v = [cv.index_select(0, idx) for cv in self.caches_v]

        # Rebuild outputs for new B
        self.logits_out = None
        self.kd_out = None
        self.vd_out = None


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--encoder-adaptor-model", type=str, required=True)
    p.add_argument("--embedding-model", type=str, required=True)
    p.add_argument("--llm-model", type=str, required=True)
    p.add_argument("--llm-tokenizer", type=str, required=True)

    p.add_argument("--wave", type=str, nargs="+", required=True)
    p.add_argument("--prompt", type=str, default="语音转写：")

    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--encoder-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--embedding-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--llm-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")

    # VAD integration
    p.add_argument("--vad-model", type=str, default="models/silero_vad.onnx",
                   help="Path to silero_vad.onnx")
    p.add_argument("--no-vad", action="store_true", help="Disable VAD; run on whole wav")
    p.add_argument("--vad-pad-sec", type=float, default=0.30, help="Pad speech segments on both sides (seconds)")
    p.add_argument("--vad-merge-gap-sec", type=float, default=0.20, help="Merge segments if gap <= this (seconds)")
    p.add_argument("--vad-min-seg-sec", type=float, default=0.20, help="Drop segments shorter than this (seconds)")
    p.add_argument("--vad-max-seg-sec", type=float, default=20.0, help="Max segment duration before requiring longer silence (seconds)")
    p.add_argument("--vad-split-overlap-sec", type=float, default=0.40,
                   help="When a segment is longer than vad-max-seg-sec, split it with this overlap (seconds)")

    # Decode optimization toggle
    p.add_argument("--compact", action="store_true",
                   help="Enable compaction (copies KV cache). Usually slower; default off.")

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

    vad_model = None
    if not args.no_vad:
        if not os.path.exists(args.vad_model):
            raise FileNotFoundError(f"--vad-model not found: {args.vad_model}")

        vad_model = SileroVAD(
            model_path=args.vad_model,
            threshold=0.5,
            min_silence_duration=0.5,
            min_speech_duration=0.25,
            window_size=512,
            max_speech_duration=20,
            sample_rate=16000,
            num_threads=1,
        )

        print(f"[VAD] model={args.vad_model}, pad={args.vad_pad_sec}s, merge_gap={args.vad_merge_gap_sec}s, min_seg={args.vad_min_seg_sec}s")
    else:
        print("[VAD] disabled; run on whole wav")

    preps: List[UtterancePrep] = []
    t0 = time.time()

    for wav in args.wave:
        samples, sr = load_and_resample_audio(wav)
        audio_dur = float(len(samples) / sr)

        segments: List[Tuple[int, int]] = []
        if args.no_vad:
            segments = [(0, int(len(samples)))]
        else:
            assert vad_model is not None
            segments = _run_vad_segments_1p1(
                vad_model=vad_model,
                samples=samples,
                sr=sr,
                pad_sec=args.vad_pad_sec,
                merge_gap_sec=args.vad_merge_gap_sec,
                min_seg_sec=args.vad_min_seg_sec,
                max_seg_sec=args.vad_max_seg_sec,
            )
            if len(segments) == 0:
                segments = [(0, int(len(samples)))]

        seg_idx = 0
        for si, (ss, ee) in enumerate(segments):
            sub_segs = _split_segment_with_overlap(
                ss, ee, sr,
                max_len_sec=args.vad_max_seg_sec,
                overlap_sec=args.vad_split_overlap_sec,
            )

            for (ss2, ee2) in sub_segs:
                seg_samples = samples[ss2:ee2]
                seg_start_sec = float(ss2 / sr)
                seg_end_sec = float(ee2 / sr)
                seg_dur_sec = seg_end_sec - seg_start_sec

                feats = compute_feat(seg_samples, sr, encoder.window_size, encoder.window_shift)[None, ...]
                enc_out = encoder(feats)
                enc_out = np.where(np.isfinite(enc_out), enc_out, 0.0)

                max_audio_frames = max(1, llm.max_total_len - 64)
                if enc_out.shape[1] > max_audio_frames:
                    enc_out = enc_out[:, :max_audio_frames, :]

                print(f"[VAD] {wav} seg{seg_idx}: {seg_dur_sec:.2f}s, frames={enc_out.shape[1]}, ss={seg_start_sec:.2f}s, ee={seg_end_sec:.2f}s")

                system_prompt = "You are a helpful assistant."
                user_prompt = f"{args.prompt}<|startofspeech|>!!<|endofspeech|>"
                audio_len = int(enc_out.shape[1])

                source_ids, fbank_beg, fake_len = build_source_ids(
                    tokenizer, system_prompt, user_prompt, audio_len
                )

                key = f"{wav}#seg{seg_idx}"
                preps.append(UtterancePrep(
                    key=key,
                    parent_wav=wav,
                    seg_start=seg_start_sec,
                    seg_end=seg_end_sec,
                    audio_dur=audio_dur,
                    source_ids=source_ids,
                    fbank_beg=fbank_beg,
                    fake_len=fake_len,
                    enc_out=enc_out
                ))
                seg_idx += 1

    by_len: Dict[int, List[UtterancePrep]] = {}
    for p in preps:
        by_len.setdefault(int(p.source_ids.shape[0]), []).append(p)

    groups: List[BatchGroup] = []

    PREFILL_BS = 1

    def _chunks(xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i+n]

    for prompt_len, items in sorted(by_len.items(), key=lambda x: x[0]):
        for sub_items in _chunks(items, PREFILL_BS):
            B = len(sub_items)
            S = int(prompt_len)

            if llm.max_total_len > 0 and S >= llm.max_total_len:
                raise RuntimeError(f"prompt_len={S} >= max_total_len={llm.max_total_len}. keys={[it.key for it in sub_items]}")

            input_ids_np = np.stack([it.source_ids for it in sub_items], axis=0).astype(np.int64, copy=False)  # [B,S]
            input_ids = torch.from_numpy(input_ids_np).to(device=device, dtype=torch.int64).contiguous()

            text_embeds = embedding.forward_ids(input_ids).to(dtype=llm.input_torch_dtype)
            text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=0.0, neginf=0.0)

            enc_stack = np.concatenate([it.enc_out for it in sub_items], axis=0)
            enc_t = torch.from_numpy(enc_stack).to(device=device, dtype=llm.input_torch_dtype).contiguous()

            for b, it in enumerate(sub_items):
                fl = int(it.fake_len)
                if fl > enc_t.shape[1]:
                    fl = int(enc_t.shape[1])
                text_embeds[b, it.fbank_beg:it.fbank_beg + fl, :] = enc_t[b, :fl, :]

            inputs_embeds = text_embeds.contiguous()

            caches_k, caches_v = llm.alloc_caches(batch=B, device=device)
            cache_position = torch.arange(0, S, device=device, dtype=torch.int64)
            attention_mask = torch.ones((B, S), device=device, dtype=torch.int64)

            logits_out = llm.alloc_logits(batch=B, device=device)
            kd_out, vd_out = llm.alloc_kv_deltas(batch=B, seq=S, device=device)

            logits, k_deltas, v_deltas = llm.run_iobinding(
                inputs_embeds, attention_mask, cache_position,
                caches_k, caches_v,
                logits_out, kd_out, vd_out
            )

            for i in range(llm.num_layers):
                caches_k[i][:, 0:S, :, :].copy_(k_deltas[i])
                caches_v[i][:, 0:S, :, :].copy_(v_deltas[i])

            g = BatchGroup(
                prompt_len=S,
                device=device,
                llm=llm,
                eos_token_id=eos_token_id,
                im_end_token_id=im_end_token_id,
                compact=args.compact,
            )
            g.caches_k = caches_k
            g.caches_v = caches_v
            g.logits_out = logits_out

            for it in sub_items:
                st = StreamState(
                    key=it.key,
                    parent_wav=it.parent_wav,
                    seg_start=it.seg_start,
                    seg_end=it.seg_end,
                    audio_dur=it.audio_dur,
                    generated=[],
                    done=False,
                )
                g.streams.append(st)

            groups.append(g)

    for g in groups:
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
                    final_results[s.key] = list(s.generated)
                continue

            any_active = True

            B = len(g.streams)
            assert g.logits_out is not None

            logits_2d = g.logits_out[:, 0, :]
            tokens = _sample_tokens_fast(
                logits_2d=logits_2d.to(dtype=torch.float32),
                temperature=args.temperature,
                top_p=args.top_p,
            )

            toks_cpu = tokens.tolist()
            for b, s in enumerate(g.streams):
                if s.done:
                    toks_cpu[b] = int(g.eos if g.eos is not None else 0)
                    continue

                tok = int(toks_cpu[b])
                s.generated.append(tok)

                if g.step > 0 and ((g.eos is not None and tok == g.eos) or (g.im_end is not None and tok == g.im_end)):
                    s.done = True
                    final_results[s.key] = list(s.generated)

            if g.compact_enabled and any(s.done for s in g.streams):
                g.compact()
                if g.active_count() == 0:
                    continue
                B = len(g.streams)
                toks_cpu = [int(g.eos if g.eos is not None else 0) for _ in range(B)]

            if g.active_count() == 0:
                continue

            B = len(g.streams)
            input_ids = torch.tensor(
                np.array(toks_cpu, dtype=np.int64).reshape(B, 1),
                device=g.device,
                dtype=torch.int64
            ).contiguous()
            tok_embeds = embedding.forward_ids(input_ids).to(dtype=llm.input_torch_dtype).contiguous()

            g.cache_position_1[0] = int(g.past_len)
            attention_mask = torch.ones((B, g.past_len + 1), device=g.device, dtype=torch.int64)

            g.ensure_decode_buffers(batch=B, seq=1)
            assert g.kd_out is not None and g.vd_out is not None and g.logits_out is not None

            _, k_deltas, v_deltas = llm.run_iobinding(
                tok_embeds, attention_mask, g.cache_position_1,
                g.caches_k, g.caches_v,
                g.logits_out, g.kd_out, g.vd_out
            )

            for i in range(llm.num_layers):
                g.caches_k[i][:, g.past_len:g.past_len + 1, :, :].copy_(k_deltas[i])
                g.caches_v[i][:, g.past_len:g.past_len + 1, :, :].copy_(v_deltas[i])

            g.past_len += 1
            g.step += 1

        if not any_active:
            break

    for g in groups:
        for s in g.streams:
            if s.key not in final_results:
                final_results[s.key] = list(s.generated)

    decode_end = time.time()
    total_time = decode_end - t0
    decode_time = decode_end - decode_start
    rtf = (total_time / total_audio) if total_audio > 0 else 0.0

    seg2text: Dict[str, str] = {}
    segmeta: Dict[str, Tuple[str, float, float]] = {}

    for g in groups:
        for s in g.streams:
            toks = final_results.get(s.key, [])
            txt = ""
            if toks:
                txt = tokenizer.decode(toks, skip_special_tokens=True)
                txt = txt.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
                txt = " ".join(txt.split())
            seg2text[s.key] = txt
            segmeta[s.key] = (s.parent_wav, s.seg_start, s.seg_end)

    wav2segs: Dict[str, List[Tuple[float, float, str]]] = {}
    for key, (parent, ss, ee) in segmeta.items():
        wav2segs.setdefault(parent, []).append((ss, ee, seg2text.get(key, "")))

    print("\n===== RESULTS =====")
    for wav in args.wave:
        segs = wav2segs.get(wav, [])
        segs.sort(key=lambda x: x[0])

        merged_txt = ""
        prev_end = None
        for (ss, ee, t) in segs:
            if not t:
                continue
            
            if prev_end is not None and ss < prev_end:
                max_k = min(200, len(merged_txt) + len(t))
            else:
                max_k = 100
            
            if not merged_txt:
                merged_txt = t
            else:
                deduped = _dedup_by_overlap(merged_txt, t, max_k=max_k)
                merged_txt += deduped
            
            prev_end = ee
        
        merged_txt = " ".join(merged_txt.split())

        print(f"\n--- {wav} ---")
        print(merged_txt)
        print()

        for i, (ss, ee, t) in enumerate(segs):
            print(f"[SEG{i}] {ss:.2f}-{ee:.2f}: {t}")

    print(f"\n[TIME] total_audio={total_audio:.3f}s, wall_total={total_time:.3f}s, wall_decode={decode_time:.3f}s, RTF={rtf:.3f}")


if __name__ == "__main__":
    main()
