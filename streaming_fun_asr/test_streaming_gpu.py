#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from streaming_fun_asr_llm import StreamingFunASRLLM, StreamingConfig, load_and_resample_audio

_default_models_dir = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    parser = argparse.ArgumentParser(description="Test StreamingFunASRLLM with a wav file")
    parser.add_argument(
        "--wave",
        type=str,
        required=True,
        help="Path to wav file"
    )
    parser.add_argument(
        "--encoder-adaptor-model",
        type=str,
        default=os.path.join(_default_models_dir, "encoder_adaptor.onnx"),
        help="Encoder adaptor model path"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.path.join(_default_models_dir, "embedding.onnx"),
        help="Embedding model path"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.path.join(_default_models_dir, "llm_fp32", "llm.fp32.onnx"),
        help="LLM model path"
    )
    parser.add_argument(
        "--llm-tokenizer",
        type=str,
        default=os.path.join(_default_models_dir, "Qwen3-0.6B"),
        help="LLM tokenizer path"
    )
    parser.add_argument(
        "--hop-ms",
        type=int,
        default=200,
        help="Hop size in milliseconds (new audio data length to feed each time, e.g., 200ms recommended)"
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=1200,
        help="Warmup duration in milliseconds (wait before outputting results)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="流式语音转写：",
        help="Prompt for streaming recognition"
    )
    args = parser.parse_args()

    if not os.path.exists(args.wave):
        print(f"Error: Wav file not found: {args.wave}")
        sys.exit(1)

    for model_name, model_path in [
        ("encoder", args.encoder_adaptor_model),
        ("embedding", args.embedding_model),
        ("llm", args.llm_model),
        ("tokenizer", args.llm_tokenizer),
    ]:
        if not os.path.exists(model_path):
            print(f"Warning: {model_name} model not found: {model_path}")

    print("=" * 80)
    print("StreamingFunASR LLM Demo")
    print("=" * 80)
    print(f"Wav file: {args.wave}")
    print(f"Encoder model: {args.encoder_adaptor_model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"LLM model: {args.llm_model}")
    print(f"Tokenizer: {args.llm_tokenizer}")
    print(f"Device: CUDA (GPU)")
    print(f"Hop size: {args.hop_ms}ms")
    print(f"Warmup: {args.warmup_ms}ms")
    print("=" * 80)

    cfg = StreamingConfig(
        encoder_adaptor_model=args.encoder_adaptor_model,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        llm_tokenizer=args.llm_tokenizer,
        runtime="gpu",
        encoder_device="cuda",
        embedding_device="cuda",
        llm_device="cuda",
        sample_rate=16000,
        prompt_zh_streaming=args.prompt,
        prompt_zh_offline="语音转写：",
        system_prompt="You are a helpful assistant.",
        temperature=0.0,
        top_p=1.0,
        max_new_tokens_per_chunk=24,
        seed=42,
        stable_drop_last_token=True,
        warmup_ms=args.warmup_ms,
        pending_chars=16,
        min_commit_chars=6,
        audio_window_ms=6000,
    )

    print("\nLoading model...")
    model = StreamingFunASRLLM(cfg)
    print("Model loaded successfully!\n")

    print(f"Loading audio file: {args.wave}")
    samples, sr = load_and_resample_audio(args.wave, target_sr=cfg.sample_rate)
    duration = len(samples) / sr
    print(f"Audio loaded: {len(samples)} samples, {sr}Hz, {duration:.2f}s duration\n")

    hop = int(cfg.sample_rate * args.hop_ms / 1000)

    print("Starting streaming recognition...")
    print("-" * 80)
    print("Recognition result (streaming):")
    print("-" * 80)

    t0 = time.time()
    full_text = ""
    last_displayed_text = ""

    for start in range(0, len(samples), hop):
        end = min(len(samples), start + hop)
        seg = samples[start:end]
        is_last = (end >= len(samples))

        seg_int16 = (seg * 32768).astype(np.int16)

        for res in model.streaming_inference(seg_int16, is_last=is_last):
            is_final = res.get("is_last", False)
            final_text = res.get("final_text", None)

            if final_text:
                text = final_text
            else:
                text = res.get("text", "")

            if text and text != last_displayed_text:
                print(f"\r{text}", end="", flush=True)
                last_displayed_text = text
                full_text = text
    
    print("\n" + "-" * 80)
    print(f"\nFinal result: {full_text}")
    print(f"\nElapsed time: {time.time() - t0:.3f}s")
    elapsed = time.time() - t0
    rtf = elapsed / duration if duration > 0 else 0
    print(f"RTF: {rtf:.2f} ({'faster' if rtf < 1.0 else 'slower'} than real-time)")
    print("=" * 80)


if __name__ == "__main__":
    main()

