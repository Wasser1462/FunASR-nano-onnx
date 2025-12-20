#!/usr/bin/env python3

import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.llm_onnx import LLMOnnx
from scripts.embedding_onnx import EmbeddingOnnx
import re


def sample_token(logits, temperature=1.0, top_p=1.0, eos_token_id=None, im_end_token_id=None, step=0):
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32)
    
    logits = np.where(np.isfinite(logits), logits, float('-inf'))

    if temperature == 0.0:
        if step == 0:
            if eos_token_id is not None:
                logits = logits.copy()
                logits[eos_token_id] = float('-inf')
            if im_end_token_id is not None:
                logits = logits.copy()
                logits[im_end_token_id] = float('-inf')
        return int(np.argmax(logits))
    
    logits = logits / temperature
    
    if step == 0:
        if eos_token_id is not None:
            logits[eos_token_id] = float('-inf')
        if im_end_token_id is not None:
            logits[im_end_token_id] = float('-inf')
    
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
                    logits[sorted_indices_to_remove] = float('-inf')
    
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


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-adaptor-model",
        type=str,
        required=True,
        help="Path to encoder+adaptor ONNX model (encoder_adaptor.onnx)",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        required=True,
        help="Path to LLM ONNX model (llm.onnx)",
    )

    parser.add_argument(
        "--llm-tokenizer",
        type=str,
        required=True,
        help="Path to HF LLM dir (same as used in export_llm_onnx.py --llm-path)",
    )

    parser.add_argument(
        "--wave",
        type=str,
        required=True,
        help="Path to audio file",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="语音转写：",
        help="User prompt text",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for sampling (higher = more random, lower = more deterministic)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling threshold",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for ONNX inference: 'cpu', 'cuda', or 'auto' (default: auto, uses CPU for LLM due to CUDA float16 issues)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Path to ONNX embedding model (embedding.onnx). If not provided, will use PyTorch model from --llm-tokenizer",
    )

    return parser.parse_args()


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_feat(
    samples: np.ndarray,
    sample_rate: int,
    window_size: int,
    window_shift: int,
):
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

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )

    T = (features.shape[0] - window_size) // window_shift + 1
    if T <= 0:
        return np.zeros((0, features.shape[1] * window_size), dtype=np.float32)

    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )

    features = np.ascontiguousarray(features, dtype=np.float32)
    return features


class EncoderAdaptorOnnxModel:
    def __init__(self, filename: str, device: str = "auto"):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        if device == "cpu":
            use_providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                use_providers = ["CPUExecutionProvider"]
        else:
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                use_providers = ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=use_providers,
        )

        meta = self.model.get_modelmeta().custom_metadata_map

        self.window_size = int(meta.get("lfr_window_size", 7))
        self.window_shift = int(meta.get("lfr_window_shift", 6))
        self.encoder_output_size = int(meta.get("encoder_output_size", 512))
        self.llm_dim = int(meta.get("llm_dim", 1024))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        encoder_out = self.model.run(
            [self.model.get_outputs()[0].name],
            {self.model.get_inputs()[0].name: x},
        )[0]
        return encoder_out


def build_source_ids_and_audio_slots(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt: str,
    audio_token_len: int,
):
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
            sub_token = tokenizer.encode(sub_str)
            source_ids += sub_token
        else:
            fake_token_len = audio_token_len
            fake_token = [0] * fake_token_len
            fbank_beg = len(source_ids)
            source_ids += fake_token

    if fbank_beg < 0:
        fbank_beg = len(source_ids)
        fake_token_len = audio_token_len
        source_ids += [0] * fake_token_len

    return np.array(source_ids, dtype=np.int64), fbank_beg, fake_token_len


def main():
    args = get_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_tokenizer,
        trust_remote_code=True,
    )

    samples, sample_rate = load_audio(args.wave)
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    if args.device == "auto":
        encoder_device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        llm_device = "cpu"
    else:
        encoder_device = args.device
        llm_device = "cpu" if args.device == "cuda" else args.device

    encoder_adaptor_model = EncoderAdaptorOnnxModel(args.encoder_adaptor_model, device=encoder_device)

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        window_size=encoder_adaptor_model.window_size,
        window_shift=encoder_adaptor_model.window_shift,
    )

    features = features[None]
    encoder_out = encoder_adaptor_model(x=features)

    llm_model = LLMOnnx(args.llm_model, device=llm_device)

    system_prompt = "You are a helpful assistant."
    user_prompt = f"{args.prompt}<|startofspeech|>!!<|endofspeech|>"

    audio_token_len = encoder_out.shape[1]

    source_ids_1d, fbank_beg_idx, fake_token_len = build_source_ids_and_audio_slots(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        audio_token_len=audio_token_len,
    )

    input_dtype = getattr(llm_model, "input_dtype", np.float32)

    # Initialize embedding layer (ONNX or PyTorch)
    use_onnx_embedding = args.embedding_model is not None
    embedding_onnx = None
    embedding_layer = None
    device = None
    
    if use_onnx_embedding:
        # print(f"Using ONNX embedding model: {args.embedding_model}")
        embedding_onnx = EmbeddingOnnx(args.embedding_model, device=llm_device)
        # Get text embeddings using ONNX
        text_embeds = embedding_onnx(source_ids_1d[None, :])
        
        if np.any(np.isnan(text_embeds)) or np.any(np.isinf(text_embeds)):
            text_embeds = np.where(np.isfinite(text_embeds), text_embeds, 0.0)
        
        if input_dtype == np.float16:
            text_embeds = np.clip(text_embeds, -65504.0, 65504.0)
            text_embeds = text_embeds.astype(input_dtype)
        else:
            text_embeds = text_embeds.astype(input_dtype)
    else:
        print(f"Using PyTorch embedding model from: {args.llm_tokenizer}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = AutoModelForCausalLM.from_pretrained(
            args.llm_tokenizer,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
        ).to(device)
        embedding_layer = embedding_model.get_input_embeddings()
        embedding_layer.eval()

        with torch.no_grad():
            input_ids_tensor = torch.from_numpy(
                source_ids_1d[None, :].astype("int64")
            ).to(device)
            text_embeds_torch = embedding_layer(input_ids_tensor)
            text_embeds = text_embeds_torch.to("cpu").numpy().astype(np.float32)
            
            if np.any(np.isnan(text_embeds)) or np.any(np.isinf(text_embeds)):
                text_embeds = np.where(np.isfinite(text_embeds), text_embeds, 0.0)
            
            if input_dtype == np.float16:
                text_embeds = np.clip(text_embeds, -65504.0, 65504.0)
                text_embeds = text_embeds.astype(input_dtype)
            else:
                text_embeds = text_embeds.astype(input_dtype)

    if np.any(np.isnan(encoder_out)) or np.any(np.isinf(encoder_out)):
        encoder_out = np.where(np.isfinite(encoder_out), encoder_out, 0.0)
    
    encoder_out = encoder_out.astype(input_dtype)

    if fake_token_len > encoder_out.shape[1]:
        fake_token_len = encoder_out.shape[1]
    if fake_token_len < encoder_out.shape[1]:
        encoder_out = encoder_out[:, :fake_token_len, :]

    inputs_embeds = text_embeds.copy()
    inputs_embeds[
        0,
        fbank_beg_idx : fbank_beg_idx + fake_token_len,
        :
    ] = encoder_out[0, :fake_token_len, :]

    inputs_embeds = np.ascontiguousarray(inputs_embeds, dtype=input_dtype)
    
    if np.any(np.isnan(inputs_embeds)) or np.any(np.isinf(inputs_embeds)):
        inputs_embeds = np.where(np.isfinite(inputs_embeds), inputs_embeds, 0.0)

    current_inputs_embeds = inputs_embeds
    valid_len = inputs_embeds.shape[1]

    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None

    generated_token_ids = []
    max_new_tokens = args.max_new_tokens
    max_total_len = 2048

    for step in range(max_new_tokens):
        if valid_len >= max_total_len:
            break

        logits = llm_model(current_inputs_embeds)
        last_idx = valid_len - 1
        next_token_logits = logits[0, last_idx, :]
        
        next_token_id = sample_token(
            next_token_logits,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            im_end_token_id=im_end_token_id,
            step=step
        )
        generated_token_ids.append(next_token_id)

        stop = False
        if step > 0:
            if eos_token_id is not None and next_token_id == eos_token_id:
                stop = True
            if im_end_token_id is not None and next_token_id == im_end_token_id:
                stop = True
        
        if stop:
            break

        # Get embedding for next token
        if use_onnx_embedding:
            next_token_embed = embedding_onnx([[next_token_id]])
            next_token_embed = next_token_embed.astype(current_inputs_embeds.dtype)
        else:
            with torch.no_grad():
                next_token_tensor = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=device
                )
                next_token_embed = (
                    embedding_layer(next_token_tensor)
                    .to("cpu")
                    .numpy()
                    .astype(current_inputs_embeds.dtype)
                )

        current_inputs_embeds = np.concatenate(
            [current_inputs_embeds, next_token_embed],
            axis=1,
        )
        valid_len += 1

    if len(generated_token_ids) > 0:
        generated_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )
        generated_text = generated_text.replace("▁", " ")
        generated_text = generated_text.replace("<|im_end|>", "")
        generated_text = generated_text.replace("<|endoftext|>", "")
        generated_text = " ".join(generated_text.split())

        print(generated_text)


if __name__ == "__main__":
    main()
