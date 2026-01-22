#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import re
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer


def pick_providers(device: str):
    providers = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]


def select_device(device_pref: str, model_path: Optional[str] = None) -> str:
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if model_path and "int8" in model_path.lower():
        return "cpu"
    return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def setup_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = int(im_end_ids[0]) if len(im_end_ids) > 0 else None
    return tokenizer, eos_token_id, im_end_token_id


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)


def load_and_resample_audio(filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    samples, sr = load_audio(filename)
    if sr != target_sr:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


# Apply Low Frame Rate (LFR) processing to reduce temporal resolution.
# Concatenates multiple consecutive frames into a single frame.
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


# Sample token from logits using temperature and top-p (nucleus) sampling.
# Handles both greedy decoding (temperature=0) and sampling.
# Returns token ID 0 as fallback if all logits are invalid.
def sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id=None,
    im_end_token_id=None,
    ban_token_ids_step0: Optional[List[int]] = None,
    step: int = 0,
) -> int:
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32)
    logits = np.where(np.isfinite(logits), logits, float("-inf"))

    if step == 0:
        logits = logits.copy()
        if eos_token_id is not None and 0 <= eos_token_id < logits.shape[0]:
            logits[eos_token_id] = float("-inf")
        if im_end_token_id is not None and 0 <= im_end_token_id < logits.shape[0]:
            logits[im_end_token_id] = float("-inf")
        if ban_token_ids_step0:
            for tid in ban_token_ids_step0:
                if 0 <= tid < logits.shape[0]:
                    logits[tid] = float("-inf")

    if temperature == 0.0:
        return int(np.argmax(logits))

    logits = logits / float(temperature)

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


def np_dtype_from_ort(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    if "int64" in s:
        return np.int64
    raise RuntimeError(f"Unsupported ORT type: {ort_type}")


def torch_dtype_from_np(np_dtype: np.dtype):
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.int64:
        return torch.int64
    raise RuntimeError(f"Unsupported numpy dtype: {np_dtype}")


def bind_torch_tensor(io: ort.IOBinding, name: str, t: torch.Tensor, is_input: bool):
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


def pick_last_logits_torch(logits_t: torch.Tensor):
    if logits_t.dim() != 3 or logits_t.shape[1] != 1:
        raise RuntimeError(f"Bad logits torch shape: {tuple(logits_t.shape)}")
    return logits_t[0, 0, :]


# Build source token IDs with chat template format:
# [system_prompt] [user_prompt] [audio_tokens] [assistant_prompt]
# Returns the token sequence and sets fbank_beg_idx to the start position
# of audio tokens in the sequence.
def build_source_ids(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str, audio_token_len: int, prev_text: str):
    pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
    source_input = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{prev_text}"
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
            # Use pad tokens as placeholders for audio embeddings
            fake_token_len = int(audio_token_len)
            fbank_beg = len(source_ids)
            source_ids += [0] * fake_token_len

    if fbank_beg < 0:
        fbank_beg = len(source_ids)
        fake_token_len = int(audio_token_len)
        source_ids += [0] * fake_token_len

    return np.array(source_ids, dtype=np.int64), int(fbank_beg), int(fake_token_len)


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
        self._io = self.sess.io_binding() if self.is_cuda else None
        out_shape = self.sess.get_outputs()[0].shape
        self.embed_dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else None
        if self.is_cuda and self.embed_dim is None:
            raise RuntimeError("Embedding output dim is dynamic; please export with static embed dim for CUDA iobinding.")

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        return self.sess.run([self.out_name], {self.in_name: input_ids})[0]

    def call_cuda_iobinding(self, input_ids_t: torch.Tensor, out_t: torch.Tensor):
        if not self.is_cuda:
            raise RuntimeError("Embedding session is not CUDA")
        io = self._io
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        io.bind_input(
            name=self.in_name,
            device_type="cuda",
            device_id=0,
            element_type=np.int64,
            shape=list(input_ids_t.shape),
            buffer_ptr=int(input_ids_t.data_ptr()),
        )
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


# LLM model with KV cache delta updates, using CUDA iobinding for efficiency.
# Model outputs are deltas that get applied in-place to the KV cache buffer.
class UnifiedKvDeltaLLMOnnxCUDA:
    def __init__(self, filename: str, device: str = "cuda"):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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
        self.out_logits = outs[0].name
        vocab_shape = outs[0].shape
        self.vocab_size = int(vocab_shape[-1]) if isinstance(vocab_shape[-1], int) else 0

        ins = {i.name: i for i in self.sess.get_inputs()}
        self.input_dtype = np_dtype_from_ort(ins["inputs_embeds"].type)
        self.cache_dtype = np_dtype_from_ort(ins["cache_key_0"].type)
        self.cache_torch_dtype = torch_dtype_from_np(self.cache_dtype)

        if self.num_layers <= 0:
            self.num_layers = len([k for k in ins.keys() if k.startswith("cache_key_")])

        self.in_inputs_embeds = "inputs_embeds"
        self.in_attention_mask = "attention_mask"
        self.in_cache_position = "cache_position"

        self.providers = self.sess.get_providers()
        self.is_cuda = ("CUDAExecutionProvider" in self.providers)
        if not self.is_cuda:
            raise RuntimeError(f"CUDAExecutionProvider not enabled for LLM session. providers={self.providers}")

        if self.vocab_size <= 0:
            raise RuntimeError("LLM logits vocab dim must be static for CUDA buffers.")

        self._io = self.sess.io_binding()

    # Create KV cache buffer [B, max_total_len, kv_h, hd].
    # This stores the accumulated KV cache. Model outputs are deltas that get applied in-place.
    def alloc_caches_t(self, batch: int = 1, device: str = "cuda"):
        if self.max_total_len <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise RuntimeError(
                f"Missing meta for cache alloc: max_total_len={self.max_total_len}, "
                f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"
            )
        caches_k_t = []
        caches_v_t = []
        for _ in range(self.num_layers):
            caches_k_t.append(torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous())
            caches_v_t.append(torch.zeros((batch, self.max_total_len, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous())
        return caches_k_t, caches_v_t

    def alloc_logits_t(self, batch: int = 1, device: str = "cuda"):
        return torch.empty((batch, 1, self.vocab_size), device=device, dtype=torch.float32).contiguous()

    def alloc_kv_deltas_t(self, batch: int, seq: int, device: str = "cuda"):
        k_delta = []
        v_delta = []
        for _ in range(self.num_layers):
            k_delta.append(torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous())
            v_delta.append(torch.empty((batch, seq, self.num_kv_heads, self.head_dim), device=device, dtype=self.cache_torch_dtype).contiguous())
        return k_delta, v_delta

    # Run LLM inference with CUDA iobinding for efficient GPU inference.
    # kv_outputs contains deltas that update cache_kv at positions specified by cache_position.
    def run_iobinding(
        self,
        inputs_embeds_t: torch.Tensor,
        attention_mask_t: torch.Tensor,
        cache_position_t: torch.Tensor,
        caches_k_t: List[torch.Tensor],
        caches_v_t: List[torch.Tensor],
        logits_out_t: torch.Tensor,
        k_delta_out_t: List[torch.Tensor],
        v_delta_out_t: List[torch.Tensor],
    ):
        io = self._io
        io.clear_binding_inputs()
        io.clear_binding_outputs()

        bind_torch_tensor(io, self.in_inputs_embeds, inputs_embeds_t, is_input=True)
        bind_torch_tensor(io, self.in_attention_mask, attention_mask_t, is_input=True)
        bind_torch_tensor(io, self.in_cache_position, cache_position_t, is_input=True)

        for i in range(self.num_layers):
            bind_torch_tensor(io, f"cache_key_{i}", caches_k_t[i], is_input=True)
            bind_torch_tensor(io, f"cache_value_{i}", caches_v_t[i], is_input=True)

        bind_torch_tensor(io, self.out_logits, logits_out_t, is_input=False)

        for i in range(self.num_layers):
            bind_torch_tensor(io, f"key_delta_{i}", k_delta_out_t[i], is_input=False)
            bind_torch_tensor(io, f"value_delta_{i}", v_delta_out_t[i], is_input=False)

        self.sess.run_with_iobinding(io)


@dataclass
class StreamingConfig:
    encoder_adaptor_model: str
    embedding_model: str
    llm_model: str
    llm_tokenizer: str
    encoder_device: str = "auto"
    embedding_device: str = "auto"
    llm_device: str = "auto"
    sample_rate: int = 16000
    prompt_zh_streaming: str = "流式语音转写："
    prompt_zh_offline: str = "语音转写："
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens_per_chunk: int = 96
    seed: int = 42
    stable_drop_last_token: bool = True
    warmup_ms: int = 1200
    pending_chars: int = 16
    min_commit_chars: int = 6
    min_commit_audio_ms: int = 2500
    audio_window_ms: int = 6000
    final_decode_on_end: bool = True
    final_max_new_tokens: int = 256
    ban_step0_strings: Tuple[str, ...] = ("A", "a", "�")


class FunASRCore:
    def __init__(self, cfg: StreamingConfig):
        self.cfg = cfg
        self.tokenizer, self.eos_token_id, self.im_end_token_id = setup_tokenizer(cfg.llm_tokenizer)

        self.enc_dev = select_device(cfg.encoder_device)
        self.emb_dev = select_device(cfg.embedding_device)
        self.llm_dev = select_device(cfg.llm_device, model_path=cfg.llm_model)

        if "int8" in cfg.llm_model.lower():
            raise RuntimeError("CUDA streaming core requires fp32/fp16 LLM model, not int8.")

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

        self.encoder = EncoderAdaptorOnnxModel(cfg.encoder_adaptor_model, device=self.enc_dev)
        self.embedding = EmbeddingOnnx(cfg.embedding_model, device=self.emb_dev)

        if self.llm_dev != "cuda":
            self.llm_dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        if self.llm_dev != "cuda":
            raise RuntimeError("CUDAExecutionProvider not available; cannot use CUDA iobinding LLM.")

        self.llm = UnifiedKvDeltaLLMOnnxCUDA(cfg.llm_model, device="cuda")

        self._ban_step0_ids: List[int] = []
        for s in cfg.ban_step0_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                self._ban_step0_ids.append(int(ids[0]))

        if self.embedding.embed_dim is None:
            raise RuntimeError("Embedding output dim must be static to run fast CUDA path.")
        self.hidden_size = int(self.embedding.embed_dim)


class FunASRSession:
    def __init__(self, core: FunASRCore):
        self.core = core
        self._alloc_cuda_buffers()
        self.reset()

    def _alloc_cuda_buffers(self):
        llm = self.core.llm
        self._caches_k_t, self._caches_v_t = llm.alloc_caches_t(batch=1, device="cuda")
        self._logits_out_t = llm.alloc_logits_t(batch=1, device="cuda")
        self._attn_mask_buf = torch.ones((1, llm.max_total_len), device="cuda", dtype=torch.int64).contiguous()
        self._k_delta_1_t, self._v_delta_1_t = llm.alloc_kv_deltas_t(batch=1, seq=1, device="cuda")

        self._k_delta_prefill_t: Optional[List[torch.Tensor]] = None
        self._v_delta_prefill_t: Optional[List[torch.Tensor]] = None
        self._prefill_seq_cached: int = -1

        if self.core.embedding.is_cuda:
            self._tok_id_t = torch.empty((1, 1), device="cuda", dtype=torch.int64)
            self._tok_embed_t = torch.empty((1, 1, self.core.hidden_size), device="cuda", dtype=torch.float32).contiguous()
        else:
            self._tok_id_t = None
            self._tok_embed_t = None

    def reset(self):
        self._all_samples = np.zeros((0,), dtype=np.float32)
        self.committed_text = ""
        self.pending_text = ""
        self.prev_text = ""
        self._last_emit_text = ""

    def _llm_clear_kv(self):
        for i in range(self.core.llm.num_layers):
            self._caches_k_t[i].zero_()
            self._caches_v_t[i].zero_()

    def _ensure_prefill_deltas(self, seq: int):
        if self._prefill_seq_cached == seq and self._k_delta_prefill_t is not None and self._v_delta_prefill_t is not None:
            return
        self._k_delta_prefill_t, self._v_delta_prefill_t = self.core.llm.alloc_kv_deltas_t(batch=1, seq=seq, device="cuda")
        self._prefill_seq_cached = seq

    def _emit_delta(self, full_text: str) -> str:
        if not self._last_emit_text:
            self._last_emit_text = full_text
            return full_text
        if full_text.startswith(self._last_emit_text):
            delta = full_text[len(self._last_emit_text):]
        else:
            delta = full_text
        self._last_emit_text = full_text
        return delta

    # Implement audio sliding window to limit _all_samples size.
    def _append_audio(self, x_f: np.ndarray):
        self._all_samples = np.concatenate([self._all_samples, x_f], axis=0)
        max_keep = int(self.core.cfg.sample_rate * self.core.cfg.audio_window_ms / 1000)
        if max_keep > 0 and self._all_samples.shape[0] > max_keep:
            self._all_samples = self._all_samples[-max_keep:]

    def _stabilize_prev_text(self, text: str) -> str:
        if not self.core.cfg.stable_drop_last_token:
            return text
        if not text:
            return ""
        toks = self.core.tokenizer.encode(text)
        if len(toks) <= 1:
            return ""
        return self.core.tokenizer.decode(toks[:-1]).replace("�", "")

    def _decode_clean(self, ids: List[int]) -> str:
        if not ids:
            return ""
        t = self.core.tokenizer.decode(ids, skip_special_tokens=True)
        t = t.replace("▁", " ").replace("<|im_end|>", "").replace("<|endoftext|>", "")
        t = " ".join(t.split())
        return t

    # Calculate the length of the longest common prefix between two strings.
    def _lcp_len(self, a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # Find a safe boundary (e.g., after punctuation) to split text.
    def _safe_boundary(self, s: str) -> int:
        if not s:
            return 0
        punct = set(" \t\r\n,.;:!?，。；：！？、）)]}》」』”’")
        for i in range(len(s) - 1, -1, -1):
            if s[i] in punct:
                return i + 1
        return 0

    # Split a given text into a committed_part and a pending_part based on pending_chars and min_commit_chars.
    def _split_commit_pending(self, text: str) -> Tuple[str, str]:
        text = text.strip()
        if not text:
            return "", ""
        cfg = self.core.cfg
        if len(text) <= cfg.pending_chars:
            return "", text
        cut = max(0, len(text) - cfg.pending_chars)
        b = self._safe_boundary(text[:cut])
        if b < cfg.min_commit_chars:
            return "", text
        return text[:b], text[b:]

    # Get text embeddings for the prompt tokens.
    # Inject audio embeddings into placeholder region (if alignment is still possible).
    def _make_inputs_embeds_np(self, encoder_out: np.ndarray, streaming: bool, prev_text_for_prompt: str):
        cfg = self.core.cfg
        audio_token_len = int(encoder_out.shape[1])
        prompt = cfg.prompt_zh_streaming if streaming else cfg.prompt_zh_offline
        user_prompt = f"{prompt}<|startofspeech|>!!<|endofspeech|>"

        source_ids_1d, fbank_beg_idx, fake_token_len = build_source_ids(
            self.core.tokenizer, cfg.system_prompt, user_prompt, audio_token_len, prev_text_for_prompt
        )

        text_embeds = self.core.embedding(source_ids_1d[None, :]).astype(np.float32)
        text_embeds = np.where(np.isfinite(text_embeds), text_embeds, 0.0)

        input_dtype = self.core.llm.input_dtype
        inputs_embeds = text_embeds.astype(input_dtype, copy=True)

        encoder_out = encoder_out.astype(input_dtype, copy=False)
        if fake_token_len > encoder_out.shape[1]:
            fake_token_len = encoder_out.shape[1]
        if fake_token_len < encoder_out.shape[1]:
            encoder_out = encoder_out[:, :fake_token_len, :]

        inputs_embeds[0, fbank_beg_idx:fbank_beg_idx + fake_token_len, :] = encoder_out[0, :fake_token_len, :]
        inputs_embeds = np.ascontiguousarray(inputs_embeds, dtype=input_dtype)

        prompt_len = int(inputs_embeds.shape[1])
        if self.core.llm.max_total_len > 0 and prompt_len >= self.core.llm.max_total_len:
            raise RuntimeError(f"prompt_len={prompt_len} >= max_total_len={self.core.llm.max_total_len}")
        return inputs_embeds, prompt_len

    # Prefill: seq = context_len, mask_len = context_len.
    # Apply KV deltas to cache buffer in-place.
    def _prefill(self, inputs_embeds_np: np.ndarray, prompt_len: int):
        llm = self.core.llm
        self._llm_clear_kv()
        self._ensure_prefill_deltas(prompt_len)

        inputs_embeds_t = torch.from_numpy(inputs_embeds_np).to(device="cuda", dtype=torch.float32, non_blocking=False).contiguous()
        attn = self._attn_mask_buf[:, :prompt_len].contiguous()
        cache_pos = torch.arange(0, prompt_len, device="cuda", dtype=torch.int64).contiguous()

        llm.run_iobinding(
            inputs_embeds_t=inputs_embeds_t,
            attention_mask_t=attn,
            cache_position_t=cache_pos,
            caches_k_t=self._caches_k_t,
            caches_v_t=self._caches_v_t,
            logits_out_t=self._logits_out_t,
            k_delta_out_t=self._k_delta_prefill_t,
            v_delta_out_t=self._v_delta_prefill_t,
        )
        torch.cuda.synchronize()

        # Apply KV deltas to cache buffer in-place.
        # kv_outputs contains deltas that update cache_kv at positions specified by cache_position.
        for i in range(llm.num_layers):
            self._caches_k_t[i][:, 0:prompt_len, :, :] = self._k_delta_prefill_t[i]
            self._caches_v_t[i][:, 0:prompt_len, :, :] = self._v_delta_prefill_t[i]

        return prompt_len

    # Decode: seq = 1, mask_len = valid_len (= past + 1).
    # Performs token generation in a loop, using CUDA iobinding for efficiency.
    def _decode_steps(self, past_len: int, max_new_tokens: int) -> List[int]:
        llm = self.core.llm
        generated: List[int] = []
        next_logits_t = pick_last_logits_torch(self._logits_out_t)

        use_emb_cuda_fast = (self.core.embedding.is_cuda and self._tok_id_t is not None and self._tok_embed_t is not None)
        cur_len = int(past_len)

        for step in range(max_new_tokens):
            if llm.max_total_len > 0 and cur_len >= llm.max_total_len:
                break

            if self.core.cfg.temperature == 0.0 and self.core.cfg.top_p >= 1.0:
                if step == 0 and (self.core.eos_token_id is not None or self.core.im_end_token_id is not None or self.core._ban_step0_ids):
                    tmp = next_logits_t.detach().clone()
                    if self.core.eos_token_id is not None:
                        tmp[self.core.eos_token_id] = -1e9
                    if self.core.im_end_token_id is not None:
                        tmp[self.core.im_end_token_id] = -1e9
                    for tid in self.core._ban_step0_ids:
                        if 0 <= tid < tmp.numel():
                            tmp[tid] = -1e9
                    tok = int(torch.argmax(tmp).item())
                else:
                    tok = int(torch.argmax(next_logits_t).item())
            else:
                logits_np = next_logits_t.detach().float().cpu().numpy()
                tok = sample_token(
                    logits_np,
                    temperature=self.core.cfg.temperature,
                    top_p=self.core.cfg.top_p,
                    eos_token_id=self.core.eos_token_id,
                    im_end_token_id=self.core.im_end_token_id,
                    ban_token_ids_step0=self.core._ban_step0_ids,
                    step=step,
                )

            generated.append(tok)

            if step > 0:
                if self.core.eos_token_id is not None and tok == self.core.eos_token_id:
                    break
                if self.core.im_end_token_id is not None and tok == self.core.im_end_token_id:
                    break

            if use_emb_cuda_fast:
                self._tok_id_t[0, 0] = int(tok)
                self.core.embedding.call_cuda_iobinding(self._tok_id_t, self._tok_embed_t)
                inputs_embeds_step_t = self._tok_embed_t
            else:
                tok_embed = self.core.embedding(np.array([[tok]], dtype=np.int64)).astype(np.float32, copy=False)
                inputs_embeds_step_t = torch.from_numpy(np.ascontiguousarray(tok_embed)).to(device="cuda", dtype=torch.float32, non_blocking=False).contiguous()

            # mask_len must equal kv_seq_len (= past + current).
            total_seq = cur_len + 1
            attn = self._attn_mask_buf[:, :total_seq].contiguous()
            cache_pos = torch.tensor([cur_len], device="cuda", dtype=torch.int64).contiguous()

            llm.run_iobinding(
                inputs_embeds_t=inputs_embeds_step_t,
                attention_mask_t=attn,
                cache_position_t=cache_pos,
                caches_k_t=self._caches_k_t,
                caches_v_t=self._caches_v_t,
                logits_out_t=self._logits_out_t,
                k_delta_out_t=self._k_delta_1_t,
                v_delta_out_t=self._v_delta_1_t,
            )
            torch.cuda.synchronize()

            # Apply KV deltas to cache buffer in-place.
            for i in range(llm.num_layers):
                self._caches_k_t[i][:, cur_len:cur_len + 1, :, :] = self._k_delta_1_t[i]
                self._caches_v_t[i][:, cur_len:cur_len + 1, :, :] = self._v_delta_1_t[i]

            cur_len += 1
            next_logits_t = pick_last_logits_torch(self._logits_out_t)

        return generated

    # Run chunk processing: prefill and decode steps, including offline final correction.
    def _run_chunk(self, encoder_out: np.ndarray, is_last: bool) -> Tuple[str, Optional[str]]:
        warmup_samples = int(self.core.cfg.sample_rate * self.core.cfg.warmup_ms / 1000)
        if len(self._all_samples) < warmup_samples and not is_last:
            return "", None

        prev_for_prompt = self._stabilize_prev_text((self.committed_text + self.pending_text).strip())

        inputs_embeds_np, prompt_len = self._make_inputs_embeds_np(
            encoder_out=encoder_out,
            streaming=not is_last,
            prev_text_for_prompt=prev_for_prompt,
        )

        past_len = self._prefill(inputs_embeds_np, prompt_len)
        ids = self._decode_steps(past_len, self.core.cfg.max_new_tokens_per_chunk)

        new_text = self._decode_clean(ids)
        full_text = (prev_for_prompt + new_text).strip()

        # Run in "offline mode" when is_last=True for a final correction.
        final_text_offline = None
        if is_last and self.core.cfg.final_decode_on_end:
            inputs_embeds2, prompt_len2 = self._make_inputs_embeds_np(
                encoder_out=encoder_out,
                streaming=False,
                prev_text_for_prompt="",
            )
            past_len2 = self._prefill(inputs_embeds2, prompt_len2)
            ids2 = self._decode_steps(past_len2, self.core.cfg.final_max_new_tokens)
            final2 = self._decode_clean(ids2).strip()
            if final2:
                final_text_offline = final2
                full_text = final2

        return full_text, final_text_offline

    def streaming_inference(self, speech_samples, is_last: bool) -> Generator[Dict, None, None]:
        if isinstance(speech_samples, torch.Tensor):
            speech_samples = speech_samples.detach().cpu().numpy()

        x = np.asarray(speech_samples)
        if x.dtype == np.int16:
            x_f = x.astype(np.float32) / 32768.0
        else:
            x_f = x.astype(np.float32)
            if np.max(np.abs(x_f)) > 1.5:
                x_f = x_f / 32768.0
        if x_f.ndim != 1:
            x_f = x_f.reshape(-1)

        self._append_audio(x_f)

        feats = compute_feat(self._all_samples, self.core.cfg.sample_rate, self.core.encoder.window_size, self.core.encoder.window_shift)
        feats = feats[None, ...]
        encoder_out = self.core.encoder(feats)
        encoder_out = np.where(np.isfinite(encoder_out), encoder_out, 0.0)

        full_text, final_text_offline = self._run_chunk(encoder_out, bool(is_last))
        if not full_text and not is_last:
            yield {"text": "", "delta": "", "is_last": bool(is_last), "timestamps": []}
            return

        # Implement commit/pending logic for stable output.
        if len(self._all_samples) < int(self.core.cfg.sample_rate * self.core.cfg.min_commit_audio_ms / 1000) and not is_last:
            self.committed_text = ""
            self.pending_text = full_text
        else:
            old_total = (self.committed_text + self.pending_text).strip()
            lcp = self._lcp_len(old_total, full_text)
            base = full_text[:lcp]
            commit_part, _ = self._split_commit_pending(base)
            self.committed_text = commit_part
            tail = full_text[len(commit_part):]
            if len(tail) > self.core.cfg.pending_chars:
                tail = tail[-self.core.cfg.pending_chars:]
            self.pending_text = tail

        final_text = (self.committed_text + self.pending_text).strip()
        self.prev_text = final_text

        yield {
            "text": final_text,
            "delta": self._emit_delta(final_text),
            "is_last": bool(is_last),
            "timestamps": [],
            "final_text": final_text_offline,
        }


class StreamingFunASRLLM:
    def __init__(self, cfg: StreamingConfig):
        self.core = FunASRCore(cfg)
        self.session = FunASRSession(self.core)

    def reset(self):
        self.session.reset()

    def streaming_inference(self, speech_samples, is_last: bool) -> Generator[Dict, None, None]:
        yield from self.session.streaming_inference(speech_samples, is_last)
