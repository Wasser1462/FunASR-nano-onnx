#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw

import argparse
import os
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

import nano_llm


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-pt",
        type=str,
        default="./model.pt",
        help="Path to model.pt file",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18, recommended. Lower versions may cause conversion errors)",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="encoder_adaptor.onnx",
        help="Output ONNX filename",
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification",
    )

    parser.add_argument(
        "--skip-verify-int8",
        action="store_true",
        help="Skip verification for int8 model",
    )

    parser.add_argument(
        "--verify-provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="ONNX Runtime provider used for verification (cpu/cuda). If cuda is unavailable, fallback to cpu",
    )

    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for fp32 ONNX verification",
    )

    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for fp32 ONNX verification",
    )

    parser.add_argument(
        "--verify-int8-tolerance",
        type=float,
        default=80.0,
        help="INT8 verification tolerance on max_abs_diff (reference=fp32 ONNX output)",
    )

    parser.add_argument(
        "--verify-int8-num-tests",
        type=int,
        default=3,
        help="Number of random tests per sequence length for INT8 verification",
    )

    parser.add_argument(
        "--export-use-dynamo",
        action="store_true",
        help="Use dynamo exporter (new exporter). Default is OFF to ensure correctness with dynamic_axes",
    )

    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def pick_ort_providers(prefer: str) -> List[str]:
    avail = ort.get_available_providers()
    if prefer == "cuda":
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[verify] CUDAExecutionProvider not available, fallback to CPU. available={avail}")
    return ["CPUExecutionProvider"]


def ort_infer(onnx_path: str, x_np: np.ndarray, providers: List[str]) -> np.ndarray:
    session_opts = ort.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(onnx_path, sess_options=session_opts, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess.run([output_name], {input_name: x_np})[0]


def assert_fp32_allclose(pt: np.ndarray, ort_out: np.ndarray, name: str, rtol: float, atol: float):
    if pt.shape != ort_out.shape:
        raise AssertionError(f"[verify] {name}: shape mismatch: pytorch={pt.shape}, onnx={ort_out.shape}")

    if not np.isfinite(pt).all():
        bad = np.argwhere(~np.isfinite(pt))
        raise AssertionError(f"[verify] {name}: PyTorch output contains NaN/Inf, first_bad_index={bad[0].tolist()}")

    if not np.isfinite(ort_out).all():
        bad = np.argwhere(~np.isfinite(ort_out))
        raise AssertionError(f"[verify] {name}: ONNX output contains NaN/Inf, first_bad_index={bad[0].tolist()}")

    diff = np.abs(pt - ort_out)
    tol = atol + rtol * np.abs(ort_out)
    ok = np.all(diff <= tol)

    max_abs = float(diff.max()) if diff.size else 0.0
    viol_cnt = int((diff > tol).sum())

    print(f"[verify] {name}: allclose={ok} rtol={rtol} atol={atol} max_abs={max_abs:.6g} viol_cnt={viol_cnt}")

    if not ok and diff.size:
        ratio = diff / (tol + 1e-12)
        worst = int(ratio.argmax())
        idx = tuple(int(x) for x in np.unravel_index(worst, ratio.shape))
        raise AssertionError(
            f"[verify] {name}: worst_violation idx={idx} "
            f"pt={pt[idx]!r} ort={ort_out[idx]!r} abs_diff={diff[idx]!r} tol={tol[idx]!r}"
        )


def int8_stats(ref: np.ndarray, out: np.ndarray):
    diff = np.abs(ref - out)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    ref_abs = np.abs(ref)
    rel = np.where(ref_abs > 1e-8, diff / (ref_abs + 1e-8), diff)
    max_rel = float(np.max(rel))
    mean_rel = float(np.mean(rel))
    return max_abs, mean_abs, max_rel, mean_rel, diff


@torch.no_grad()
def verify_fp32(model: torch.nn.Module, onnx_path: str, x: torch.Tensor, providers: List[str], rtol: float, atol: float, name: str):
    model.eval()
    x_cpu = x.detach().cpu().to(torch.float32)

    pt_out = model(x_cpu)
    if isinstance(pt_out, (tuple, list)):
        pt_out = pt_out[0]
    pt_np = pt_out.detach().cpu().numpy()

    ort_out = ort_infer(onnx_path, x_cpu.numpy().astype(np.float32), providers)
    assert_fp32_allclose(pt_np, ort_out, name=name, rtol=rtol, atol=atol)


def verify_int8_like_embedding(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    x_list: List[Tuple[str, torch.Tensor]],
    providers: List[str],
    tolerance: float,
    num_tests: int,
):
    print("[verify] Verifying INT8 model (reference=fp32 ONNX output)...")

    all_passed = True
    worst = {
        "max_abs": 0.0,
        "tag": "",
        "test_idx": -1,
        "idx": None,
        "ref": None,
        "out": None,
    }

    for tag, x in x_list:
        for test_idx in range(num_tests):
            x_cpu = x.detach().cpu().to(torch.float32)
            x_np = x_cpu.numpy().astype(np.float32)

            ref = ort_infer(fp32_onnx_path, x_np, providers)
            out = ort_infer(int8_onnx_path, x_np, providers)

            max_abs, mean_abs, max_rel, mean_rel, diff = int8_stats(ref, out)
            passed = max_abs < tolerance
            status = "PASS" if passed else "FAIL"

            print(f"[verify] int8({tag}) test {test_idx + 1}/{num_tests}: {status}")
            print(f"  max_abs_diff={max_abs:.6g}  mean_abs_diff={mean_abs:.6g}")
            print(f"  max_rel_diff={max_rel:.6g}  mean_rel_diff={mean_rel:.6g}  tol={tolerance:.6g}")

            if max_abs > worst["max_abs"]:
                idx = np.unravel_index(int(diff.argmax()), diff.shape)
                worst["max_abs"] = max_abs
                worst["tag"] = tag
                worst["test_idx"] = test_idx
                worst["idx"] = tuple(int(i) for i in idx)
                worst["ref"] = float(ref[idx])
                worst["out"] = float(out[idx])

            if not passed:
                all_passed = False

    print("[verify] INT8 Summary:")
    print(f"  overall={'PASSED' if all_passed else 'FAILED'}")
    print(f"  worst_max_abs_diff={worst['max_abs']:.6g} at {worst['tag']} test {worst['test_idx'] + 1}/{num_tests}")
    if worst["idx"] is not None:
        print(f"  worst_idx={worst['idx']} ref(fp32_onnx)={worst['ref']:.6g} out(int8)={worst['out']:.6g} tol={tolerance:.6g}")

    if not all_passed:
        raise AssertionError("[verify] INT8 verification failed (max_abs_diff exceeds tolerance).")


def export_onnx(model: torch.nn.Module, x: torch.Tensor, filename: str, opset_version: int, use_dynamo: bool):
    print(f"Exporting to ONNX (opset {opset_version})...")

    export_kwargs = dict(
        opset_version=opset_version,
        input_names=["x"],
        output_names=["encoder_out"],
        dynamic_axes={
            "x": {1: "T"},
            "encoder_out": {1: "T_out"},
        },
        verbose=False,
        do_constant_folding=True,
        external_data=False,
    )

    sig = inspect.signature(torch.onnx.export)
    if "training" in sig.parameters:
        export_kwargs["training"] = torch.onnx.TrainingMode.EVAL
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = bool(use_dynamo)

    torch.onnx.export(model, x, filename, **export_kwargs)


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    if not Path(args.model_pt).exists():
        raise ValueError(f"Model file not found: {args.model_pt}")

    # Load state dict first to detect adaptor layer count
    data = torch.load(args.model_pt, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        state_dict = data
    
    # Detect adaptor layer count from state_dict
    adaptor_block_keys = [k for k in state_dict.keys() if k.startswith("audio_adaptor.blocks.")]
    adaptor_blocks = set()
    for k in adaptor_block_keys:
        parts = k.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            adaptor_blocks.add(int(parts[2]))
    
    n_layer = len(adaptor_blocks) if adaptor_blocks else 0
    print(f"Detected adaptor layers: {n_layer} (blocks: {sorted(adaptor_blocks) if adaptor_blocks else 'none'})")
    
    # Create model with correct number of layers
    adaptor_config = {
        "downsample_rate": 1,
        "encoder_dim": 512,  # Default for SenseVoiceEncoderSmall
        "llm_dim": 1024,    # Default for Qwen3-0.6B
        "ffn_dim": 2048,
        "n_layer": n_layer,
    }
    
    print("Loading model...")
    model = nano_llm.NanoLLM(adaptor_config=adaptor_config)

    # Filter state dict to only include encoder and adaptor
    # Note: model.pt uses "audio_adaptor." prefix, but model expects "adaptor." prefix
    encoder_adaptor_dict = {}
    for key, value in state_dict.items():
        if key.startswith("audio_encoder."):
            encoder_adaptor_dict[key] = value
        elif key.startswith("audio_adaptor."):
            # Remove "audio_" prefix to match model's expected key names
            new_key = key.replace("audio_adaptor.", "adaptor.", 1)
            encoder_adaptor_dict[new_key] = value

    if len(encoder_adaptor_dict) == 0:
        print("Warning: No encoder/adaptor parameters found!")
        print("Available keys (first 20):")
        for key in list(state_dict.keys())[:20]:
            print(f"  {key}")
        raise ValueError("Failed to load encoder+adaptor parameters")

    missing_keys, unexpected_keys = model.load_state_dict(encoder_adaptor_dict, strict=False)
    model.eval()
    
    # Convert model to float32 for ONNX export (model.pt uses bfloat16)
    # This ensures consistency with ONNX Runtime which typically uses float32
    model = model.float()

    print(f"Loaded {len(encoder_adaptor_dict)} parameters for encoder+adaptor")
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys not found in model (first 5):")
        for key in missing_keys[:5]:
            print(f"    {key}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys (first 5):")
        for key in unexpected_keys[:5]:
            print(f"    {key}")
    if not missing_keys and not unexpected_keys:
        print("All parameters loaded successfully!")
    print(" Model converted to float32 for ONNX export")

    # Create dummy input (audio features)
    # Shape: (batch, time, 560) - 560 = 80 * 7 (LFR features)
    x = torch.randn(1, 30, 560, dtype=torch.float32)

    opset_version = args.opset_version
    filename = args.output_filename

    export_onnx(
        model=model,
        x=x,
        filename=filename,
        opset_version=opset_version,
        use_dynamo=args.export_use_dynamo,
    )

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)

    model_author = "FunAudioLLM"
    comment = os.environ.get("comment", "FunAudioLLM/Fun-ASR-Nano-2512")
    url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512"

    meta_data = {
        "lfr_window_size": 7,
        "lfr_window_shift": 6,
        "normalize_samples": 0,
        "model_type": "sense_voice_encoder_adaptor",
        "version": "1",
        "model_author": model_author,
        "encoder_output_size": model.encoder_output_size,
        "llm_dim": model.llm_dim,
        "comment": comment,
        "url": url,
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    print(f"ONNX model saved to: {filename}")

    providers = pick_ort_providers(args.verify_provider)
    if not args.skip_verify:
        print(f"[verify] Using providers: {providers}")
        test_inputs: List[Tuple[str, torch.Tensor]] = [
            ("T=30", x),
            ("T=47", torch.randn(1, 47, 560, dtype=torch.float32)),
        ]
        for tag, x_test in test_inputs:
            verify_fp32(
                model=model,
                onnx_path=filename,
                x=x_test,
                providers=providers,
                rtol=args.verify_rtol,
                atol=args.verify_atol,
                name=f"fp32({tag})",
            )

    filename_int8 = filename.replace(".onnx", ".int8.onnx")
    print("Quantizing to INT8...")
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"Quantized model saved to: {filename_int8}")

    if (not args.skip_verify) and (not args.skip_verify_int8):
        test_inputs_int8: List[Tuple[str, torch.Tensor]] = [
            ("T=30", torch.randn(1, 30, 560, dtype=torch.float32)),
            ("T=47", torch.randn(1, 47, 560, dtype=torch.float32)),
        ]
        verify_int8_like_embedding(
            fp32_onnx_path=filename,
            int8_onnx_path=filename_int8,
            x_list=test_inputs_int8,
            providers=providers,
            tolerance=args.verify_int8_tolerance,
            num_tests=args.verify_int8_num_tests,
        )


if __name__ == "__main__":
    torch.manual_seed(20260114)
    main()

