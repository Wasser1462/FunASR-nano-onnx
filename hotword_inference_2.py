#!/usr/bin/env python3
import argparse
import logging
import re
import time
from typing import Tuple, List, Optional

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class HotwordBiaser:
    def __init__(self, tokenizer, hotwords: List[str], boost: float = 10.0, start_boost: float = 1.0, start_steps: int = 6):
        self.hotwords = [w for w in hotwords if w and w.strip()]
        self.seqs = [tokenizer.encode(w, add_special_tokens=False) for w in self.hotwords]
        self.boost = float(boost)
        self.start_boost = float(start_boost)
        self.start_steps = int(start_steps)

    def debug_dump(self, tokenizer):
        for w, s in zip(self.hotwords, self.seqs):
            print(f"[hotword] {w} => ids={s} => decode='{tokenizer.decode(s)}'")

    def apply(self, logits: np.ndarray, history_ids: List[int], step: int):
        if not self.seqs:
            return

        if step < self.start_steps:
            for seq in self.seqs:
                if len(seq) > 0:
                    logits[seq[0]] += self.start_boost

        for seq in self.seqs:
            L = len(seq)
            if L <= 1:
                continue

            max_k = min(len(history_ids), L - 1)
            best = 0
            for k in range(max_k, 0, -1):
                if history_ids[-k:] == seq[:k]:
                    best = k
                    break

            if best > 0 and best < L:
                next_tid = seq[best]
                logits[next_tid] += self.boost


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
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)


def compute_feat(samples: np.ndarray, sample_rate: int, window_size: int, window_shift: int) -> np.ndarray:
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

    features = np.stack([online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)])  # [F,80]

    T = (features.shape[0] - window_size) // window_shift + 1
    if T <= 0:
        return np.zeros((0, features.shape[1] * window_size), dtype=np.float32)

    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    return np.ascontiguousarray(features, dtype=np.float32)


def infer_downsample_k(lfr_len: int, out_len: int, max_k: int = 64) -> Optional[int]:
    for k in range(1, max_k + 1):
        if (lfr_len - 1) // k + 1 == out_len:
            return k
    return None


class EncoderAdaptorOnnxModel:
    def __init__(self, filename: str, device: str = "auto", lfr_window_shift: int = 6):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(filename, sess_options=so, providers=pick_providers(device))

        inp = self.sess.get_inputs()[0]
        in_shape = inp.shape
        c = None
        if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 3 and isinstance(in_shape[2], int):
            c = int(in_shape[2])
        if c is None:
            c = 560
        if c % 80 != 0:
            raise RuntimeError(f"encoder_adaptor input dim C={c} not divisible by 80")
        self.window_size = c // 80
        self.window_shift = int(lfr_window_shift)

        self.in_name = inp.name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out_name], {self.in_name: x})[0]


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
        caches_k, caches_v = [], []
        for _ in range(self.num_layers):
            caches_k.append(np.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), dtype=self.cache_dtype))
            caches_v.append(np.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), dtype=self.cache_dtype))
        return caches_k, caches_v

    def run(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        cache_position: np.ndarray,
        caches_k: List[np.ndarray],
        caches_v: List[np.ndarray],
    ):
        feed = {
            self.in_inputs_embeds: np.ascontiguousarray(inputs_embeds, dtype=self.input_dtype),
            self.in_attention_mask: np.ascontiguousarray(attention_mask, dtype=np.int64),
            self.in_cache_position: np.ascontiguousarray(cache_position, dtype=np.int64),
        }
        for i in range(self.num_layers):
            feed[f"cache_key_{i}"] = np.ascontiguousarray(caches_k[i], dtype=self.cache_dtype)
            feed[f"cache_value_{i}"] = np.ascontiguousarray(caches_v[i], dtype=self.cache_dtype)
        return self.sess.run(None, feed)


def parse_hotwords(hotwords: Optional[str], hotwords_file: Optional[str]) -> List[str]:
    words: List[str] = []
    if hotwords_file:
        with open(hotwords_file, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.append(w)
    if hotwords:
        for w in re.split(r"[,\s]+", hotwords.strip()):
            w = w.strip()
            if w:
                words.append(w)
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def get_prompt(hotwords: List[str], language: Optional[str] = None, itn: bool = True) -> str:
    if len(hotwords) > 0:
        hotwords_s = ", ".join(hotwords)
        prompt = (
            "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
            "**上下文信息：**\n\n\n"
        )
        prompt += f"热词列表：[{hotwords_s}]\n"
    else:
        prompt = ""
    if language is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{language}"
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："


def build_source_ids(tokenizer: AutoTokenizer, system_prompt: str, user_prompt_with_speech: str, audio_token_len: int):
    pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
    source_input = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt_with_speech}<|im_end|>\n"
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
    p.add_argument("--embedding-model", type=str, required=True)
    p.add_argument("--llm-model", type=str, required=True)
    p.add_argument("--llm-tokenizer", type=str, required=True)
    p.add_argument("--wave", type=str, required=True)

    p.add_argument("--language", type=str, default=None)
    p.add_argument("--hotwords", type=str, default=None)
    p.add_argument("--hotwords-file", type=str, default=None)
    try:
        p.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True)
    except Exception:
        p.add_argument("--itn", action="store_true", default=True)
        p.add_argument("--no-itn", dest="itn", action="store_false")
    p.add_argument("--prompt", type=str, default=None)

    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--encoder-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--llm-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--embedding-device", type=str, choices=["cpu", "cuda", "auto"], default="auto")

    p.add_argument("--lfr-window-shift", type=int, default=6)
    p.add_argument("--infer-k-max", type=int, default=64)
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

    hotwords_list = parse_hotwords(args.hotwords, args.hotwords_file)

    biaser = HotwordBiaser(tokenizer, hotwords_list, boost=12.0, start_boost=1.5, start_steps=8)
    biaser.debug_dump(tokenizer)

    prompt = args.prompt if args.prompt is not None else get_prompt(hotwords_list, args.language, args.itn)

    samples, sr = load_audio(args.wave)
    if sr != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio_duration = len(samples) / sr
    start_time = time.time()

    encoder = EncoderAdaptorOnnxModel(args.encoder_adaptor_model, device=enc_dev, lfr_window_shift=args.lfr_window_shift)

    feats = compute_feat(samples, sr, encoder.window_size, encoder.window_shift)
    feats = feats[None, ...]
    lfr_len = int(feats.shape[1])

    encoder_out = encoder(feats)
    encoder_out = np.where(np.isfinite(encoder_out), encoder_out, 0.0)
    audio_token_len = int(encoder_out.shape[1])

    k_guess = infer_downsample_k(lfr_len, audio_token_len, max_k=int(args.infer_k_max))
    logging.info(
        f"[DownsampleCheck] lfr_len={lfr_len} (LFR≈/{encoder.window_shift}), encoder_out_len={audio_token_len}, inferred_adaptor_k={k_guess}"
    )

    system_prompt = "You are a helpful assistant."
    user_prompt = f"{prompt}<|startofspeech|>!!<|endofspeech|>"
    logging.info(f"[user_prompt]\n{user_prompt}")
    if hotwords_list:
        logging.info(f"[hotword_ids]={tokenizer.encode(hotwords_list[0], add_special_tokens=False)}")

    source_ids_1d, fbank_beg_idx, fake_token_len = build_source_ids(tokenizer, system_prompt, user_prompt, audio_token_len)

    embedding = EmbeddingOnnx(args.embedding_model, device=emb_dev)
    text_embeds = embedding(source_ids_1d[None, :]).astype(np.float32)
    text_embeds = np.where(np.isfinite(text_embeds), text_embeds, 0.0)

    llm = UnifiedKvDeltaLLMOnnx(args.llm_model, device=llm_dev)
    logging.info(f"[LLM] device={llm_dev}, quant={llm.quant_type}, input_dtype={llm.input_dtype}, cache_dtype={llm.cache_dtype}")
    logging.info(f"[LLM] layers={llm.num_layers}, max_total_len={llm.max_total_len}, kv_heads={llm.num_kv_heads}, head_dim={llm.head_dim}")

    input_dtype = llm.input_dtype
    inputs_embeds = text_embeds.astype(input_dtype, copy=True)

    encoder_out = encoder_out.astype(input_dtype, copy=False)
    fake_token_len = int(min(fake_token_len, encoder_out.shape[1]))
    if fake_token_len < encoder_out.shape[1]:
        encoder_out = encoder_out[:, :fake_token_len, :]

    inputs_embeds[0, fbank_beg_idx : fbank_beg_idx + fake_token_len, :] = encoder_out[0, :fake_token_len, :]
    inputs_embeds = np.ascontiguousarray(inputs_embeds, dtype=input_dtype)

    prompt_len = int(inputs_embeds.shape[1])
    logging.info(f"[prompt] prompt_len={prompt_len}, audio_token_len={audio_token_len}, fake_token_len={fake_token_len}")

    if llm.max_total_len > 0 and prompt_len >= llm.max_total_len:
        raise RuntimeError(f"prompt_len={prompt_len} >= max_total_len={llm.max_total_len}")

    caches_k, caches_v = llm.alloc_caches(batch=1)

    cache_position = np.arange(0, prompt_len, dtype=np.int64)
    attention_mask = np.ones((1, prompt_len), dtype=np.int64)

    outs = llm.run(inputs_embeds, attention_mask, cache_position, caches_k, caches_v)
    logits = outs[0]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected logits shape: {getattr(logits, 'shape', None)}")

    logits_s = int(logits.shape[1])
    if logits_s != prompt_len and logits_s != 1:
        logging.info(f"[warn] unexpected logits_seq_len={logits_s} (prompt_len={prompt_len})")

    delta_s = None
    if llm.num_layers > 0:
        k0 = outs[1]
        if k0.ndim >= 2:
            delta_s = int(k0.shape[1])

    if logits_s != prompt_len:
        logging.info(f"[warn] logits_seq_len={logits_s} != prompt_len={prompt_len} (prefill returns last-step logits)")

    for i in range(llm.num_layers):
        k_delta = outs[1 + 2 * i]
        v_delta = outs[1 + 2 * i + 1]
        ks = int(k_delta.shape[1])
        if ks == prompt_len:
            caches_k[i][:, :prompt_len, :, :] = k_delta.astype(llm.cache_dtype, copy=False)
            caches_v[i][:, :prompt_len, :, :] = v_delta.astype(llm.cache_dtype, copy=False)
        elif ks == 1:
            pos = prompt_len - 1
            caches_k[i][:, pos : pos + 1, :, :] = k_delta.astype(llm.cache_dtype, copy=False)
            caches_v[i][:, pos : pos + 1, :, :] = v_delta.astype(llm.cache_dtype, copy=False)
        else:
            m = min(prompt_len, ks)
            caches_k[i][:, :m, :, :] = k_delta[:, :m, :, :].astype(llm.cache_dtype, copy=False)
            caches_v[i][:, :m, :, :] = v_delta[:, :m, :, :].astype(llm.cache_dtype, copy=False)

    if logits_s == prompt_len:
        next_logits = logits[0, prompt_len - 1, :]
    else:
        next_logits = logits[0, -1, :]

    past_len = prompt_len

    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None

    generated: List[int] = []

    for step in range(args.max_new_tokens):
        if llm.max_total_len > 0 and past_len >= llm.max_total_len:
            break
        biaser.apply(next_logits, generated, step)
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
        logits_step = outs[0]
        if logits_step.ndim != 3 or logits_step.shape[0] != 1 or logits_step.shape[1] != 1:
            raise RuntimeError(f"Unexpected step logits shape: {getattr(logits_step, 'shape', None)}")

        for i in range(llm.num_layers):
            k_delta = outs[1 + 2 * i]
            v_delta = outs[1 + 2 * i + 1]
            caches_k[i][:, past_len : past_len + 1, :, :] = k_delta.astype(llm.cache_dtype, copy=False)
            caches_v[i][:, past_len : past_len + 1, :, :] = v_delta.astype(llm.cache_dtype, copy=False)

        past_len += 1
        next_logits = logits_step[0, 0, :]
        if np.any(~np.isfinite(next_logits)):
            next_logits = np.where(np.isfinite(next_logits), next_logits, -1e9).astype(np.float32, copy=False)

    processing_time = time.time() - start_time
    rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

    if generated:
        out_text = tokenizer.decode(generated, skip_special_tokens=True)
        out_text = out_text.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
        out_text = " ".join(out_text.split())
        print(out_text)

    print(f"[RTF] audio_duration={audio_duration:.3f}s, processing_time={processing_time:.3f}s, RTF={rtf:.3f}")


if __name__ == "__main__":
    main()
