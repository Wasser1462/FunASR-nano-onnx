#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import argparse
import re
import time
from typing import Tuple, List

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer


def sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id=None,
    im_end_token_id=None,
    step: int = 0,
) -> int:
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32)

    logits = np.where(np.isfinite(logits), logits, float("-inf"))

    if temperature == 0.0:
        if step == 0:
            logits = logits.copy()
            if eos_token_id is not None:
                logits[eos_token_id] = float("-inf")
            if im_end_token_id is not None:
                logits[im_end_token_id] = float("-inf")
        return int(np.argmax(logits))

    logits = logits / float(temperature)

    if step == 0:
        logits = logits.copy()
        if eos_token_id is not None:
            logits[eos_token_id] = float("-inf")
        if im_end_token_id is not None:
            logits[im_end_token_id] = float("-inf")

    if top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        max_logit = np.max(sorted_logits)
        if np.isfinite(max_logit):
            exp_logits = np.exp(sorted_logits - max_logit)
            cumulative_probs = np.cumsum(exp_logits)
            if cumulative_probs[-1] > 0:
                cumulative_probs = cumulative_probs / cumulative_probs[-1]
                sorted_indices_to_remove = sorted_indices[cumulative_probs > top_p]
                if len(sorted_indices_to_remove) > 0:
                    sorted_indices_to_remove = sorted_indices_to_remove[1:]
                    logits = logits.copy()
                    logits[sorted_indices_to_remove] = float("-inf")

    max_logit = np.max(logits)
    if not np.isfinite(max_logit):
        probs = np.ones_like(logits) / len(logits)
    else:
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits)
        if not np.isfinite(sum_exp) or sum_exp <= 0:
            probs = np.ones_like(logits) / len(logits)
        else:
            probs = exp_logits / sum_exp
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                probs = np.ones_like(logits) / len(logits)

    token_id = np.random.choice(len(probs), p=probs)
    return int(token_id)


def pick_providers(device: str):
    providers = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    # auto
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)


def compute_feat(samples: np.ndarray, sample_rate: int, window_size: int, window_shift: int):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    if online_fbank.num_frames_ready == 0:
        return np.zeros((0, 80 * window_size), dtype=np.float32)

    features = np.stack([online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)])

    T = (features.shape[0] - window_size) // window_shift + 1
    if T <= 0:
        return np.zeros((0, features.shape[1] * window_size), dtype=np.float32)

    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    return np.ascontiguousarray(features, dtype=np.float32)


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

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        return self.sess.run([self.out_name], {self.in_name: input_ids})[0]


def _np_dtype_from_ort(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    if "int64" in s:
        return np.int64
    raise RuntimeError(f"Unsupported ORT type: {ort_type}")


class UnifiedKvDeltaLLMOnnx:
    def __init__(self, filename: str, device: str = "cpu"):
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

        ins = {i.name: i for i in self.sess.get_inputs()}
        self.input_dtype = _np_dtype_from_ort(ins["inputs_embeds"].type)
        self.cache_dtype = _np_dtype_from_ort(ins["cache_key_0"].type)

        if self.num_layers <= 0:
            self.num_layers = len([k for k in ins.keys() if k.startswith("cache_key_")])

        self.in_inputs_embeds = "inputs_embeds"
        self.in_attention_mask = "attention_mask"
        self.in_cache_position = "cache_position"

    def alloc_caches(self, batch: int = 1):
        if self.max_total_len <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise RuntimeError(
                f"Missing meta for cache alloc: max_total_len={self.max_total_len}, "
                f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"
            )
        caches_k = []
        caches_v = []
        for _ in range(self.num_layers):
            caches_k.append(np.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), dtype=self.cache_dtype))
            caches_v.append(np.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), dtype=self.cache_dtype))
        return caches_k, caches_v

    def run(self, inputs_embeds: np.ndarray, attention_mask: np.ndarray, cache_position: np.ndarray,
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

def build_source_ids(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str, audio_token_len: int):
    pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

    source_input = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    splits = pattern.split(source_input)

    source_ids = []
    fbank_beg = -1
    fake_token_len = 0

    for sub_str in splits:
        if not sub_str:
            continue
        if not sub_str.startswith("<|startofspeech|>"):
            source_ids += tokenizer.encode(sub_str)
        else:
            fake_token_len = int(audio_token_len)
            fbank_beg = len(source_ids)
            source_ids += [0] * fake_token_len

    if fbank_beg < 0:
        fbank_beg = len(source_ids)
        fake_token_len = int(audio_token_len)
        source_ids += [0] * fake_token_len

    return np.array(source_ids, dtype=np.int64), int(fbank_beg), int(fake_token_len)


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--encoder-adaptor-model", type=str, required=True)
    p.add_argument("--embedding-model", type=str, required=True, help="Same-source embedding.onnx")
    p.add_argument("--llm-model", type=str, required=True, help="unified kv-delta llm onnx (llm.int8.onnx / llm.fp32.onnx)")
    p.add_argument("--llm-tokenizer", type=str, required=True)

    p.add_argument("--wave", type=str, required=True)
    p.add_argument("--prompt", type=str, default="语音转写：")

    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--encoder-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--llm-device", type=str, choices=["cpu", "cuda", "auto"], default="auto",
                   help="int8 动态量化建议 cpu；auto 会对 int8 强制 cpu")
    p.add_argument("--embedding-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def main():
    args = get_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_tokenizer, trust_remote_code=True)

    enc_dev = args.encoder_device
    if enc_dev == "auto":
        enc_dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"

    emb_dev = args.embedding_device
    if emb_dev == "auto":
        emb_dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"

    llm_dev = args.llm_device
    if llm_dev == "auto":
        llm_dev = "cpu" if "int8" in args.llm_model.lower() else ("cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu")

    samples, sr = load_audio(args.wave)
    if sr != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        sr = 16000

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
    print(f"[LLM] device={llm_dev}, quant={llm.quant_type}, input_dtype={llm.input_dtype}, cache_dtype={llm.cache_dtype}")
    print(f"[LLM] layers={llm.num_layers}, max_total_len={llm.max_total_len}, kv_heads={llm.num_kv_heads}, head_dim={llm.head_dim}")

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
    print(f"[prompt] prompt_len={prompt_len}, audio_token_len={audio_token_len}, fake_token_len={fake_token_len}")

    if llm.max_total_len > 0 and prompt_len >= llm.max_total_len:
        raise RuntimeError(f"prompt_len={prompt_len} >= max_total_len={llm.max_total_len}，请增大导出 max_total_len 或缩短音频/提示")

    caches_k, caches_v = llm.alloc_caches(batch=1)

    cache_position = np.arange(0, prompt_len, dtype=np.int64)
    attention_mask = np.ones((1, prompt_len), dtype=np.int64)
    outs = llm.run(inputs_embeds, attention_mask, cache_position, caches_k, caches_v)

    logits = outs[0]  # [1,S,vocab]
    for i in range(llm.num_layers):
        k_delta = outs[1 + 2 * i]       # [1,S,kv,hd]
        v_delta = outs[1 + 2 * i + 1]
        caches_k[i][:, 0:prompt_len, :, :] = k_delta.astype(llm.cache_dtype, copy=False)
        caches_v[i][:, 0:prompt_len, :, :] = v_delta.astype(llm.cache_dtype, copy=False)

    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None

    generated: List[int] = []
    next_logits = logits[0, prompt_len - 1, :]
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

        outs = llm.run(tok_embed, attention_mask, cache_position, caches_k, caches_v)
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
