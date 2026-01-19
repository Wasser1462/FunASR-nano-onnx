#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
# Common utility functions for inference and batch inference

import re
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer


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


def select_device(device_pref: str, model_path: str = None) -> str:
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    # auto
    if model_path and "int8" in model_path.lower():
        return "cpu"
    return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def setup_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None
    return tokenizer, eos_token_id, im_end_token_id


def load_and_resample_audio(filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    samples, sr = load_audio(filename)
    if sr != target_sr:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return samples, sr

