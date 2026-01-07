#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCausalLM, AutoConfig


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--llm-config-path",
        type=str,
        required=True,
        help="Path to LLM config directory (e.g., Qwen3-0.6B)",
    )
    parser.add_argument(
        "--model-pt",
         type=str, 
         default=None,
        help="Path to trained checkpoint/model.pt. If set, export embedding from this checkpoint."
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="embedding.onnx",
        help="Output ONNX filename",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Dummy sequence length for export",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported ONNX model",
    )
    return parser.parse_args()


def check_safetensors_file(llm_config_path: str) -> bool:
    config_path = Path(llm_config_path)
    model_files = list(config_path.glob("*.safetensors"))
    if model_files:
        print(f"Found safetensors file(s): {[str(f) for f in model_files]}")
        return True
    if (config_path / "model.safetensors.index.json").exists():
        print(f"Found sharded safetensors model: {config_path / 'model.safetensors.index.json'}")
        return True
    if (config_path / "pytorch_model.bin").exists():
        print(f"Found pytorch_model.bin (fallback)")
        return True
    print(f"Warning: No safetensors or pytorch_model.bin found in {config_path}")
    print(f"Available files: {list(config_path.glob('*'))}")
    return False


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)

def load_checkpoint_state_dict(model_pt: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(model_pt, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    fixed = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        fixed[nk] = v
    return fixed


def find_embedding_weight(
    state_dict: Dict[str, torch.Tensor],
    vocab_size: int = None,
    hidden_size: int = None,
) -> torch.Tensor:

    patterns = [
        "embed_tokens.weight",        
        "tok_embeddings.weight", 
        "word_embeddings.weight", 
        "wte.weight",
        "embeddings.word_embeddings.weight",
    ]

    candidates = []
    for k, v in state_dict.items():
        if not (torch.is_tensor(v) and v.ndim == 2):
            continue
        lk = k.lower()
        if any(p in lk for p in patterns):
            candidates.append((k, v))

    if vocab_size is not None and hidden_size is not None:
        for k, v in candidates:
            if v.shape[0] == vocab_size and v.shape[1] == hidden_size:
                return v
        for k, v in candidates:
            if v.shape[0] == vocab_size:
                return v

    all_2d = [(k, v) for k, v in state_dict.items() if torch.is_tensor(v) and v.ndim == 2]
    if vocab_size is not None:
        shape_match = [(k, v) for k, v in all_2d if v.shape[0] == vocab_size]
        if shape_match:
            shape_match.sort(key=lambda kv: kv[1].shape[1], reverse=True)
            return shape_match[0][1]

    if all_2d:
        all_2d.sort(key=lambda kv: kv[1].shape[0], reverse=True)
        return all_2d[0][1]

    raise RuntimeError("No 2D tensor found in checkpoint; cannot locate embedding weight.")


@torch.no_grad()
def export_embedding_onnx(
    embedding_layer: torch.nn.Module,
    vocab_size: int,
    hidden_size: int,
    filename: str,
    seq_length: int,
    opset_version: int,
):
    embedding_layer.eval()
    batch_size = 1
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.int64)
    
    print(f"Exporting embedding layer to ONNX...")
    print(f"  Input shape: {dummy_input_ids.shape}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output shape: ({batch_size}, {seq_length}, {hidden_size})")
    
    os.environ["TORCH_ONNX_DISABLE_DYNAMO"] = "1"
    torch.onnx.export(
        embedding_layer,
        dummy_input_ids,
        filename,
        opset_version=opset_version,
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size", 1: "sequence_length"},
        },
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        external_data=False,
    )
    print(f"ONNX model saved to: {filename}")


def verify_onnx_model(
    onnx_filename: str,
    llm_config_path: str,
    model_pt: str = None,
    seq_length: int = 100,
    num_tests: int = 3,
):
    print("Verifying ONNX embedding model...")

    if model_pt is not None:
        print(f"Loading embedding weight from checkpoint for verification: {model_pt}")
        config = AutoConfig.from_pretrained(llm_config_path, trust_remote_code=True)
        vocab_size = getattr(config, "vocab_size", None)
        hidden_size = getattr(config, "hidden_size", None)
        sd = load_checkpoint_state_dict(model_pt)
        w = find_embedding_weight(sd, vocab_size=vocab_size, hidden_size=hidden_size)
        w = w.detach().to(torch.float32).cpu()
        pytorch_embedding = torch.nn.Embedding(w.shape[0], w.shape[1])
        pytorch_embedding.weight.data.copy_(w)
        pytorch_embedding.eval()
    else:
        print("Loading PyTorch HF model for verification (fallback)...")
        pytorch_model = AutoModelForCausalLM.from_pretrained(
            llm_config_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
        )
        pytorch_embedding = pytorch_model.get_input_embeddings()
        pytorch_embedding.eval()
    
    print("Loading ONNX model...")
    session_opts = ort.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1
    
    try:
        ort_session = ort.InferenceSession(
            onnx_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False
    
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"ONNX output name: {output_name}")
    
    all_passed = True
    max_diff = 0.0
    max_relative_diff = 0.0
    
    for test_idx in range(num_tests):
        vocab_size = pytorch_embedding.num_embeddings
        input_ids = torch.randint(0, vocab_size, (1, seq_length), dtype=torch.int64)
        
        with torch.no_grad():
            pytorch_output = pytorch_embedding(input_ids).numpy()
        
        onnx_output = ort_session.run(
            [output_name],
            {input_name: input_ids.numpy()},
        )[0]
        
        diff = np.abs(pytorch_output - onnx_output)
        max_abs_diff = np.max(diff)
        mean_abs_diff = np.mean(diff)
        
        pytorch_abs = np.abs(pytorch_output)
        relative_diff = np.where(
            pytorch_abs > 1e-8,
            diff / (pytorch_abs + 1e-8),
            diff
        )
        max_rel_diff = np.max(relative_diff)
        mean_rel_diff = np.mean(relative_diff)
        
        max_diff = max(max_diff, max_abs_diff)
        max_relative_diff = max(max_relative_diff, max_rel_diff)
        
        tolerance = 1e-5
        passed = max_abs_diff < tolerance
        
        status = "PASS" if passed else "FAIL"
        print(f"\nTest {test_idx + 1}/{num_tests}: {status}")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {onnx_output.shape}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        
        if not passed:
            all_passed = False
            print(f"Warning: Difference exceeds tolerance ({tolerance})")
    
    print("Verification Summary:")
    print(f"Overall status: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Max absolute difference across all tests: {max_diff:.2e}")
    print(f"Max relative difference across all tests: {max_relative_diff:.2e}")
    
    if all_passed:
        print("\nAll tests passed! ONNX model matches PyTorch model.")
    else:
        print("\nSome tests failed. Please check the differences.")
    
    return all_passed


def quantize_embedding_model(model, filename, filename_int8, original_size):
    op_types = set(node.op_type for node in model.graph.node)
    print(f"Operations in model: {op_types}")

    if "Gather" not in op_types:
        print("Quantizing to INT8...")
        quantize_dynamic(
            model_input=filename,
            model_output=filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QUInt8,
        )
        quantized_size = os.path.getsize(filename_int8) / (1024 * 1024)
        size_reduction = original_size - quantized_size
        print(f"Quantized model saved to: {filename_int8}")
        print(f"Quantized model size: {quantized_size:.2f}MB (reduced by {size_reduction:.2f}MB, {size_reduction/original_size*100:.1f}%)")
        return True

    print("Embedding model uses Gather, quantizing embedding weights...")
    gather_node = None
    embedding_weight_name = None

    for node in model.graph.node:
        if node.op_type == "Gather":
            gather_node = node
            embedding_weight_name = node.input[0]
            break

    if gather_node is None:
        print("Warning: Gather node not found")
        shutil.copy(filename, filename_int8)
        return False

    for initializer in model.graph.initializer:
        if initializer.name == embedding_weight_name and len(initializer.dims) >= 2:
            weight_array = numpy_helper.to_array(initializer)
            original_size_mb = weight_array.nbytes / (1024 * 1024)

            weight_min = np.min(weight_array)
            weight_max = np.max(weight_array)

            if weight_max <= weight_min:
                continue

            scale = (weight_max - weight_min) / 255.0
            zero_point = np.clip(np.round(-weight_min / scale), 0, 255).astype(np.uint8)
            quantized_weight = np.clip(
                np.round((weight_array - weight_min) / scale), 0, 255
            ).astype(np.uint8)

            quantized_size_mb = quantized_weight.nbytes / (1024 * 1024)
            saved_mb = original_size_mb - quantized_size_mb

            initializer.data_type = onnx.TensorProto.UINT8
            initializer.raw_data = quantized_weight.tobytes()

            scale_name = initializer.name + "_scale"
            zp_name = initializer.name + "_zero_point"

            scale_tensor = onnx.helper.make_tensor(scale_name, onnx.TensorProto.FLOAT, [], [float(scale)])
            zp_tensor = onnx.helper.make_tensor(zp_name, onnx.TensorProto.UINT8, [], [int(zero_point)])

            model.graph.initializer.append(scale_tensor)
            model.graph.initializer.append(zp_tensor)

            gather_output_name = gather_node.output[0]
            intermediate_output = gather_output_name + "_uint8"

            gather_node.output[0] = intermediate_output

            dequant_node = onnx.helper.make_node(
                "DequantizeLinear",
                inputs=[intermediate_output, scale_name, zp_name],
                outputs=[gather_output_name],
                name="DequantizeLinear_0"
            )

            node_idx = list(model.graph.node).index(gather_node)
            model.graph.node.insert(node_idx + 1, dequant_node)

            print(f" Quantized {initializer.name}: {initializer.dims}, saved {saved_mb:.2f}MB")

            try:
                onnx.checker.check_model(model)
                onnx.save(model, filename_int8)
                quantized_size = os.path.getsize(filename_int8) / (1024 * 1024)
                size_reduction = original_size - quantized_size
                print(f"Quantized model saved to: {filename_int8}")
                print(f"Quantized model size: {quantized_size:.2f}MB (reduced by {size_reduction:.2f}MB, {size_reduction/original_size*100:.1f}%)")
                return True
            except Exception as e:
                print(f"Error saving quantized model: {e}")
                print("Falling back to original model")
                shutil.copy(filename, filename_int8)
                return False

    print("Warning: No weights were quantized, copying original model")
    shutil.copy(filename, filename_int8)
    return False


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))
    
    llm_config_path = Path(args.llm_config_path)
    if not llm_config_path.exists():
        raise ValueError(f"LLM config path not found: {llm_config_path}")
    
    print(f"\nChecking for model files in {llm_config_path}...")
    has_model = check_safetensors_file(str(llm_config_path))
    if not has_model:
        print("Warning: No model files found. Model loading may fail.")
    
    print(f"\nLoading LLM config from {llm_config_path}...")
    config = AutoConfig.from_pretrained(str(llm_config_path), trust_remote_code=True)
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    print(f"Model config: vocab_size={vocab_size}, hidden_size={hidden_size}")

    print(f"\nLoading LLM config from {llm_config_path}...")
    config = AutoConfig.from_pretrained(str(llm_config_path), trust_remote_code=True)
    vocab_size = getattr(config, "vocab_size", None)
    hidden_size = getattr(config, "hidden_size", None)
    print(f"Model config: vocab_size={vocab_size}, hidden_size={hidden_size}")

    if args.model_pt is not None:
        print(f"\nLoading embedding weight from checkpoint: {args.model_pt}")
        sd = load_checkpoint_state_dict(args.model_pt)
        w = find_embedding_weight(sd, vocab_size=vocab_size, hidden_size=hidden_size)
        w = w.detach().to(torch.float32).cpu()

        print(f"Found embedding weight shape: {tuple(w.shape)}")
        embedding_layer = torch.nn.Embedding(num_embeddings=w.shape[0], embedding_dim=w.shape[1])
        embedding_layer.weight.data.copy_(w)
        embedding_layer.eval()
    else:
        print(f"\nLoading PyTorch model from HF directory (fallback)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_config_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
        )
        embedding_layer = model.get_input_embeddings()
        embedding_layer.eval()
        embedding_layer.to("cpu")

    vocab_size = embedding_layer.weight.shape[0]
    hidden_size = embedding_layer.weight.shape[1]
    print(f"Export target: vocab_size={vocab_size}, hidden_size={hidden_size}") 
    print(f"Embedding layer: {type(embedding_layer).__name__}")
    print(f"  Weight shape: {embedding_layer.weight.shape}")
    print(f"  Num embeddings: {embedding_layer.num_embeddings}")
    print(f"  Embedding dim: {embedding_layer.embedding_dim}")
    
    seq_length = args.seq_length
    opset_version = args.opset_version
    filename = args.output_filename
    
    export_embedding_onnx(
        embedding_layer=embedding_layer,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        filename=filename,
        seq_length=seq_length,
        opset_version=opset_version,
    )
    
    model_author = "FunAudioLLM"
    comment = os.environ.get("comment", "FunAudioLLM/Fun-ASR-Nano-2512")
    url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512"
    meta_data = {
        "model_type": "embedding_layer",
        "version": "1",
        "model_author": model_author,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "comment": comment,
        "url": url,
    }
    add_meta_data(filename, meta_data)
    print("Metadata added to ONNX model.")
    
    filename_int8 = filename.replace(".onnx", ".int8.onnx")
    model = onnx.load(filename)
    original_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"Original model size: {original_size:.2f}MB")

    quantize_embedding_model(model, filename, filename_int8, original_size)
    add_meta_data(filename_int8, meta_data)
    
    if args.verify:
        verify_onnx_model(
            onnx_filename=filename,
            llm_config_path=str(llm_config_path),
            seq_length=min(seq_length, 200),
            num_tests=5,
        )
    else:
        print("\nNote: Use --verify flag to verify the exported model")


if __name__ == "__main__":
    torch.manual_seed(20251219)
    main()

