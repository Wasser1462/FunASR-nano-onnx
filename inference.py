#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import argparse
import time
from typing import List

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
        out = self.sess.run([self.out_name], {self.in_name: x})[0]
        return out


class EmbeddingOnnx:
    def __init__(self, filename: str, device: str = "cpu"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        self.providers = self.sess.get_providers()
        self.is_cuda = ("CUDAExecutionProvider" in self.providers)

        # optional iobinding for cuda (decode step)
        self._io = self.sess.io_binding() if self.is_cuda else None

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        return self.sess.run([self.out_name], {self.in_name: input_ids})[0]

    # CUDA iobinding path for single token embedding (fast path)
    def call_cuda_iobinding(self, input_ids_t: torch.Tensor, out_t: torch.Tensor):
        if not self.is_cuda:
            raise RuntimeError("Embedding session is not CUDA")
        if input_ids_t.dtype != torch.int64 or not input_ids_t.is_cuda:
            raise RuntimeError("input_ids_t must be CUDA int64")
        if out_t.dtype not in (torch.float16, torch.float32) or not out_t.is_cuda:
            raise RuntimeError("out_t must be CUDA float")

        io = self._io
        io.clear_binding_inputs()
        io.clear_binding_outputs()

        # bind input ids
        io.bind_input(
            name=self.in_name,
            device_type="cuda",
            device_id=0,
            element_type=np.int64,
            shape=list(input_ids_t.shape),
            buffer_ptr=int(input_ids_t.data_ptr()),
        )

        # bind output embeds
        elem = np.float16 if out_t.dtype == torch.float16 else np.float32
        io.bind_output(
            name=self.out_name,
            device_type="cuda",
            device_id=0,
            element_type=elem,
            shape=list(out_t.shape),
            buffer_ptr=int(out_t.data_ptr()),
        )

        self.sess.run_with_iobinding(io)


def _np_dtype_from_ort(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    if "int64" in s:
        return np.int64
    raise RuntimeError(f"Unsupported ORT type: {ort_type}")


def _torch_dtype_from_np(dt: np.dtype):
    if dt == np.float16:
        return torch.float16
    if dt == np.float32:
        return torch.float32
    if dt == np.int64:
        return torch.int64
    raise RuntimeError(f"Unsupported numpy dtype: {dt}")


def _bind_torch_tensor(io: ort.IOBinding, name: str, t: torch.Tensor, is_input: bool):
    if not t.is_cuda:
        raise RuntimeError(f"Tensor for '{name}' must be CUDA tensor")
    if not t.is_contiguous():
        t = t.contiguous()

    if t.dtype == torch.float16:
        elem = np.float16
    elif t.dtype == torch.float32:
        elem = np.float32
    elif t.dtype == torch.int64:
        elem = np.int64
    else:
        raise RuntimeError(f"Unsupported torch dtype for binding: {t.dtype}")

    if is_input:
        io.bind_input(
            name=name,
            device_type="cuda",
            device_id=0,
            element_type=elem,
            shape=list(t.shape),
            buffer_ptr=int(t.data_ptr()),
        )
    else:
        io.bind_output(
            name=name,
            device_type="cuda",
            device_id=0,
            element_type=elem,
            shape=list(t.shape),
            buffer_ptr=int(t.data_ptr()),
        )
    return t


def _pick_last_logits_np(logits: np.ndarray, prompt_len: int):
    # Support both old ([B,S,V]) and new ([B,1,V]) export.
    if logits.ndim != 3:
        raise RuntimeError(f"Bad logits ndim={logits.ndim}, shape={logits.shape}")
    if logits.shape[1] == 1:
        return logits[0, 0, :]
    # legacy
    return logits[0, prompt_len - 1, :]


def _pick_last_logits_torch(logits_t: torch.Tensor):
    # logits is expected to be [B,1,V]
    if logits_t.dim() != 3 or logits_t.shape[1] != 1:
        raise RuntimeError(f"Bad logits torch shape: {tuple(logits_t.shape)}")
    return logits_t[0, 0, :]


class UnifiedKvDeltaLLMOnnx:
    def __init__(self, filename: str, device: str = "cpu"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        # speed: max optimization
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # NOTE: iobinding + dynamic shapes: mem_pattern may cause trouble on some ORT builds.
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False

        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))
        meta = self.sess.get_modelmeta().custom_metadata_map

        self.quant_type = str(meta.get("quantization_type", ""))
        self.num_layers = int(meta.get("num_layers", 0) or 0)
        self.max_total_len = int(meta.get("max_total_len", 0) or 0)
        self.num_kv_heads = int(meta.get("num_kv_heads", 0) or 0)
        self.head_dim = int(meta.get("head_dim", 0) or 0)

        outs = self.sess.get_outputs()
        # logits output is always present, and should be [B,1,V] for fast export
        self.out_logits = outs[0].name

        ins = {i.name: i for i in self.sess.get_inputs()}
        self.input_dtype = _np_dtype_from_ort(ins["inputs_embeds"].type)
        self.cache_dtype = _np_dtype_from_ort(ins["cache_key_0"].type)

        if self.num_layers <= 0:
            self.num_layers = len([k for k in ins.keys() if k.startswith("cache_key_")])

        self.in_inputs_embeds = "inputs_embeds"
        self.in_attention_mask = "attention_mask"
        self.in_cache_position = "cache_position"

        self.providers = self.sess.get_providers()
        self.is_cuda = ("CUDAExecutionProvider" in self.providers)

        # iobinding only used when CUDA
        self._io = self.sess.io_binding() if self.is_cuda else None

        # cache torch dtype
        self.cache_torch_dtype = _torch_dtype_from_np(self.cache_dtype)

    def alloc_caches(self, batch: int = 1):
        if self.max_total_len <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise RuntimeError(
                f"Missing meta for cache alloc: max_total_len={self.max_total_len}, "
                f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"
            )
        if not self.is_cuda:
            caches_k = []
            caches_v = []
            for _ in range(self.num_layers):
                caches_k.append(np.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), dtype=self.cache_dtype))
                caches_v.append(np.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), dtype=self.cache_dtype))
            return caches_k, caches_v

        # CUDA: allocate caches on GPU (torch)
        caches_k_t = []
        caches_v_t = []
        for _ in range(self.num_layers):
            caches_k_t.append(torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), device="cuda", dtype=self.cache_torch_dtype))
            caches_v_t.append(torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), device="cuda", dtype=self.cache_torch_dtype))
        return caches_k_t, caches_v_t

    # CPU path: keep original behavior (sess.run + numpy)
    def run_cpu(self, inputs_embeds: np.ndarray, attention_mask: np.ndarray, cache_position: np.ndarray,
                caches_k: List[np.ndarray], caches_v: List[np.ndarray]):
        feed = {
            self.in_inputs_embeds: np.ascontiguousarray(inputs_embeds, dtype=self.input_dtype),
            self.in_attention_mask: np.ascontiguousarray(attention_mask, dtype=np.int64),
            self.in_cache_position: np.ascontiguousarray(cache_position, dtype=np.int64),
        }
        for i in range(self.num_layers):
            feed[f"cache_key_{i}"] = np.ascontiguousarray(caches_k[i], dtype=self.cache_dtype)
            feed[f"cache_value_{i}"] = np.ascontiguousarray(caches_v[i], dtype=self.cache_dtype)
        return self.sess.run(None, feed)

    # CUDA path: iobinding (inputs/outputs all on GPU)
    def run_cuda_iobinding(
        self,
        inputs_embeds_t: torch.Tensor,         # [B,S,H] float32 cuda
        attention_mask_t: torch.Tensor,        # [B,total] int64 cuda
        cache_position_t: torch.Tensor,        # [S] int64 cuda
        caches_k_t: List[torch.Tensor],        # each [B,max,kv,hd] cuda
        caches_v_t: List[torch.Tensor],
        logits_out_t: torch.Tensor,            # [B,1,V] float32 cuda
        k_delta_out_t: List[torch.Tensor],     # each [B,S,kv,hd] cuda
        v_delta_out_t: List[torch.Tensor],
    ):
        if not self.is_cuda:
            raise RuntimeError("run_cuda_iobinding called on non-cuda session")

        io = self._io
        io.clear_binding_inputs()
        io.clear_binding_outputs()

        _bind_torch_tensor(io, self.in_inputs_embeds, inputs_embeds_t, is_input=True)
        _bind_torch_tensor(io, self.in_attention_mask, attention_mask_t, is_input=True)
        _bind_torch_tensor(io, self.in_cache_position, cache_position_t, is_input=True)

        for i in range(self.num_layers):
            _bind_torch_tensor(io, f"cache_key_{i}", caches_k_t[i], is_input=True)
            _bind_torch_tensor(io, f"cache_value_{i}", caches_v_t[i], is_input=True)

        _bind_torch_tensor(io, self.out_logits, logits_out_t, is_input=False)

        for i in range(self.num_layers):
            _bind_torch_tensor(io, f"key_delta_{i}", k_delta_out_t[i], is_input=False)
            _bind_torch_tensor(io, f"value_delta_{i}", v_delta_out_t[i], is_input=False)

        self.sess.run_with_iobinding(io)

    def run(self, inputs_embeds, attention_mask, cache_position, caches_k, caches_v):
        # Keep a unified API for compatibility
        if not self.is_cuda:
            return self.run_cpu(inputs_embeds, attention_mask, cache_position, caches_k, caches_v)
        raise RuntimeError("Use run_cuda_iobinding for CUDA session")




def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--encoder-adaptor-model", type=str, required=True)
    p.add_argument("--embedding-model", type=str, required=True, help="Same-source embedding.onnx")
    p.add_argument("--llm-model", type=str, required=True, help="unified kv-delta llm onnx (llm.int8.onnx / llm.fp32.onnx / llm.fp16.onnx)")
    p.add_argument("--llm-tokenizer", type=str, required=True)

    p.add_argument("--wave", type=str, required=True)
    p.add_argument("--prompt", type=str, default="语音转写：")

    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--encoder-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--llm-device", type=str, choices=["cpu", "cuda", "auto"], default="auto",
                   help="int8 dynamic quantization recommended on cpu; auto will force cpu for int8")
    p.add_argument("--embedding-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def main():
    args = get_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    tokenizer, eos_token_id, im_end_token_id = setup_tokenizer(args.llm_tokenizer)

    enc_dev = select_device(args.encoder_device)
    emb_dev = select_device(args.embedding_device)
    llm_dev = select_device(args.llm_device, model_path=args.llm_model)

    # int8 recommended on cpu to avoid misuse
    if "int8" in args.llm_model.lower() and llm_dev != "cpu":
        print("[WARN] int8 llm model is recommended to run on CPU. force llm_device=cpu")
        llm_dev = "cpu"

    samples, sr = load_and_resample_audio(args.wave)

    audio_duration = len(samples) / sr
    start_time = time.time()

    encoder = EncoderAdaptorOnnxModel(args.encoder_adaptor_model, device=enc_dev)
    feats = compute_feat(samples, sr, encoder.window_size, encoder.window_shift)
    feats = feats[None, ...]
    encoder_out = encoder(feats)  # [1, audio_len, llm_dim]
    encoder_out = np.where(np.isfinite(encoder_out), encoder_out, 0.0)

    system_prompt = "You are a helpful assistant."
    user_prompt = f"{args.prompt}<|startofspeech|>!!<|endofspeech|>"
    audio_token_len = int(encoder_out.shape[1])

    source_ids_1d, fbank_beg_idx, fake_token_len = build_source_ids(
        tokenizer, system_prompt, user_prompt, audio_token_len
    )

    embedding = EmbeddingOnnx(args.embedding_model, device=emb_dev)
    text_embeds = embedding(source_ids_1d[None, :]).astype(np.float32)
    text_embeds = np.where(np.isfinite(text_embeds), text_embeds, 0.0)

    llm = UnifiedKvDeltaLLMOnnx(args.llm_model, device=llm_dev)
    print(f"[LLM] device={llm_dev}, providers={llm.providers}, quant={llm.quant_type}, input_dtype={llm.input_dtype}, cache_dtype={llm.cache_dtype}")
    print(f"[LLM] layers={llm.num_layers}, max_total_len={llm.max_total_len}, kv_heads={llm.num_kv_heads}, head_dim={llm.head_dim}")
    print(f"[EMB] device={emb_dev}, providers={embedding.providers}")

    input_dtype = llm.input_dtype
    inputs_embeds = text_embeds.astype(input_dtype, copy=True)

    encoder_out = encoder_out.astype(input_dtype, copy=False)
    if fake_token_len > encoder_out.shape[1]:
        fake_token_len = encoder_out.shape[1]
    if fake_token_len < encoder_out.shape[1]:
        encoder_out = encoder_out[:, :fake_token_len, :]

    inputs_embeds[0, fbank_beg_idx:fbank_beg_idx + fake_token_len, :] = encoder_out[0, :fake_token_len, :]
    inputs_embeds = np.ascontiguousarray(inputs_embeds, dtype=input_dtype)

    prompt_len = int(inputs_embeds.shape[1])
    hidden_size = int(inputs_embeds.shape[2])
    print(f"[prompt] prompt_len={prompt_len}, audio_token_len={audio_token_len}, fake_token_len={fake_token_len}")

    if llm.max_total_len > 0 and prompt_len >= llm.max_total_len:
        raise RuntimeError(f"prompt_len={prompt_len} >= max_total_len={llm.max_total_len}, please increase max_total_len during export or shorten audio/prompt")

    generated: List[int] = []

    if not llm.is_cuda:
        caches_k, caches_v = llm.alloc_caches(batch=1)

        cache_position = np.arange(0, prompt_len, dtype=np.int64)
        attention_mask = np.ones((1, prompt_len), dtype=np.int64)

        outs = llm.run_cpu(inputs_embeds, attention_mask, cache_position, caches_k, caches_v)

        logits = outs[0]  # new: [1,1,vocab]; old: [1,S,vocab]
        for i in range(llm.num_layers):
            k_delta = outs[1 + 2 * i]       # [1,S,kv,hd]
            v_delta = outs[1 + 2 * i + 1]
            caches_k[i][:, 0:prompt_len, :, :] = k_delta.astype(llm.cache_dtype, copy=False)
            caches_v[i][:, 0:prompt_len, :, :] = v_delta.astype(llm.cache_dtype, copy=False)

        next_logits = _pick_last_logits_np(logits, prompt_len=prompt_len)
        past_len = prompt_len

        for step in range(args.max_new_tokens):
            if llm.max_total_len > 0 and past_len >= llm.max_total_len:
                break

            tok = sample_token(
                next_logits,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_token_id,
                im_end_token_id=im_end_token_id,
                step=step,
            )
            generated.append(tok)

            if step > 0:
                if eos_token_id is not None and tok == eos_token_id:
                    break
                if im_end_token_id is not None and tok == im_end_token_id:
                    break

            tok_embed = embedding(np.array([[tok]], dtype=np.int64)).astype(input_dtype, copy=False)
            tok_embed = np.ascontiguousarray(tok_embed, dtype=input_dtype)

            cache_position = np.array([past_len], dtype=np.int64)
            attention_mask = np.ones((1, past_len + 1), dtype=np.int64)

            outs = llm.run_cpu(tok_embed, attention_mask, cache_position, caches_k, caches_v)
            logits_step = outs[0]  # [1,1,vocab]

            for i in range(llm.num_layers):
                k_delta = outs[1 + 2 * i]       # [1,1,kv,hd]
                v_delta = outs[1 + 2 * i + 1]
                caches_k[i][:, past_len:past_len + 1, :, :] = k_delta.astype(llm.cache_dtype, copy=False)
                caches_v[i][:, past_len:past_len + 1, :, :] = v_delta.astype(llm.cache_dtype, copy=False)

            past_len += 1
            next_logits = logits_step[0, 0, :]
            if np.any(~np.isfinite(next_logits)):
                next_logits = np.where(np.isfinite(next_logits), next_logits, -1e9).astype(np.float32, copy=False)

    else:
        # Allocate caches on GPU
        caches_k_t, caches_v_t = llm.alloc_caches(batch=1)

        # Prefill: allocate outputs on GPU
        # logits: always fp32 [1,1,vocab]
        logits_out_t = torch.empty((1, 1, int(llm.sess.get_outputs()[0].shape[-1] or 0)), device="cuda", dtype=torch.float32)
        if logits_out_t.shape[-1] == 0:
            # fallback: infer vocab from runtime output after first run (rare)
            logits_out_t = torch.empty((1, 1, 1), device="cuda", dtype=torch.float32)

        # delta buffers for prefill: [1,S,kv,hd]
        k_delta_t: List[torch.Tensor] = []
        v_delta_t: List[torch.Tensor] = []
        for _ in range(llm.num_layers):
            k_delta_t.append(torch.empty((1, prompt_len, llm.num_kv_heads, llm.head_dim), device="cuda", dtype=llm.cache_torch_dtype))
            v_delta_t.append(torch.empty((1, prompt_len, llm.num_kv_heads, llm.head_dim), device="cuda", dtype=llm.cache_torch_dtype))

        # inputs to GPU
        inputs_embeds_t = torch.from_numpy(inputs_embeds).to(device="cuda", dtype=torch.float32, non_blocking=False).contiguous()
        attention_mask_buf = torch.ones((1, llm.max_total_len if llm.max_total_len > 0 else (prompt_len + args.max_new_tokens + 2)), device="cuda", dtype=torch.int64)
        attention_mask_t = attention_mask_buf[:, :prompt_len].contiguous()
        cache_position_t = torch.arange(0, prompt_len, device="cuda", dtype=torch.int64).contiguous()

        # Prefill run
        llm.run_cuda_iobinding(
            inputs_embeds_t=inputs_embeds_t,
            attention_mask_t=attention_mask_t,
            cache_position_t=cache_position_t,
            caches_k_t=caches_k_t,
            caches_v_t=caches_v_t,
            logits_out_t=logits_out_t,
            k_delta_out_t=k_delta_t,
            v_delta_out_t=v_delta_t,
        )
        torch.cuda.synchronize()

        # Write deltas into caches on GPU
        for i in range(llm.num_layers):
            caches_k_t[i][:, 0:prompt_len, :, :] = k_delta_t[i]
            caches_v_t[i][:, 0:prompt_len, :, :] = v_delta_t[i]

        past_len = prompt_len

        # For greedy sampling (temperature==0 and top_p==1), do argmax on GPU (no cpu copy)
        def greedy_tok_from_logits_t(logits_vec_t: torch.Tensor, step: int) -> int:
            # logits_vec_t: [V] on CUDA
            if step == 0 and (eos_token_id is not None or im_end_token_id is not None):
                tmp = logits_vec_t.detach().clone()
                if eos_token_id is not None:
                    tmp[eos_token_id] = -1e9
                if im_end_token_id is not None:
                    tmp[im_end_token_id] = -1e9
                return int(torch.argmax(tmp).item())
            return int(torch.argmax(logits_vec_t).item())

        next_logits_t = _pick_last_logits_torch(logits_out_t)

        # Optional: embedding CUDA iobinding for 1-token decode
        use_emb_cuda_fast = (embedding.is_cuda and llm.is_cuda)
        tok_id_t = torch.empty((1, 1), device="cuda", dtype=torch.int64) if use_emb_cuda_fast else None
        tok_embed_t = torch.empty((1, 1, hidden_size), device="cuda", dtype=torch.float32) if use_emb_cuda_fast else None

        # Decode: allocate delta buffers of S=1 and reuse
        k_delta_1_t: List[torch.Tensor] = []
        v_delta_1_t: List[torch.Tensor] = []
        for _ in range(llm.num_layers):
            k_delta_1_t.append(torch.empty((1, 1, llm.num_kv_heads, llm.head_dim), device="cuda", dtype=llm.cache_torch_dtype))
            v_delta_1_t.append(torch.empty((1, 1, llm.num_kv_heads, llm.head_dim), device="cuda", dtype=llm.cache_torch_dtype))

        for step in range(args.max_new_tokens):
            if llm.max_total_len > 0 and past_len >= llm.max_total_len:
                break

            # sample token
            if args.temperature == 0.0 and args.top_p >= 1.0:
                tok = greedy_tok_from_logits_t(next_logits_t, step=step)
            else:
                # fallback: copy logits to cpu and use existing sampler
                next_logits_np = next_logits_t.detach().float().cpu().numpy()
                tok = sample_token(
                    next_logits_np,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=eos_token_id,
                    im_end_token_id=im_end_token_id,
                    step=step,
                )

            generated.append(tok)

            if step > 0:
                if eos_token_id is not None and tok == eos_token_id:
                    break
                if im_end_token_id is not None and tok == im_end_token_id:
                    break

            # Build tok embedding
            if use_emb_cuda_fast:
                tok_id_t[0, 0] = int(tok)
                embedding.call_cuda_iobinding(tok_id_t, tok_embed_t)
                inputs_embeds_step_t = tok_embed_t
            else:
                tok_embed = embedding(np.array([[tok]], dtype=np.int64)).astype(np.float32, copy=False)
                inputs_embeds_step_t = torch.from_numpy(np.ascontiguousarray(tok_embed)).to(device="cuda", dtype=torch.float32, non_blocking=False)

            # attention_mask slice
            total_seq = past_len + 1
            attention_mask_t = attention_mask_buf[:, :total_seq].contiguous()
            cache_position_t = torch.tensor([past_len], device="cuda", dtype=torch.int64)

            # Run decode step (S=1)
            llm.run_cuda_iobinding(
                inputs_embeds_t=inputs_embeds_step_t,
                attention_mask_t=attention_mask_t,
                cache_position_t=cache_position_t,
                caches_k_t=caches_k_t,
                caches_v_t=caches_v_t,
                logits_out_t=logits_out_t,
                k_delta_out_t=k_delta_1_t,
                v_delta_out_t=v_delta_1_t,
            )
            torch.cuda.synchronize()

            # Update caches at [past_len:past_len+1)
            for i in range(llm.num_layers):
                caches_k_t[i][:, past_len:past_len + 1, :, :] = k_delta_1_t[i]
                caches_v_t[i][:, past_len:past_len + 1, :, :] = v_delta_1_t[i]

            past_len += 1
            next_logits_t = _pick_last_logits_torch(logits_out_t)

    end_time = time.time()
    processing_time = end_time - start_time

    rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

    if generated:
        out_text = tokenizer.decode(generated, skip_special_tokens=True)
        out_text = out_text.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
        out_text = " ".join(out_text.split())
        print(out_text)

    print(f"[RTF] audio_duration={audio_duration:.3f}s, processing_time={processing_time:.3f}s, RTF={rtf:.3f}")


if __name__ == "__main__":
    main()
