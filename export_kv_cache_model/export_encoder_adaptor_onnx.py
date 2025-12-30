#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import onnx
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

    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


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

    # Warn if opset version is too low
    if opset_version < 18:
        print(f"Warning: opset_version {opset_version} is lower than recommended (18).")
        print("PyTorch may automatically upgrade to opset 18, which can cause conversion errors.")
        print("Consider using --opset-version 18 to avoid version conversion issues.")
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        x,
        filename,
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
    
    # Check actual opset version of exported model
    import onnx
    onnx_model = onnx.load(filename)
    actual_opset = onnx_model.opset_import[0].version
    if actual_opset != opset_version:
        print(f"Note: Model was exported with opset {actual_opset} (requested {opset_version})")
        print(f"This is normal - PyTorch uses the best compatible opset version.")

    # Check actual opset version of exported model
    import onnx
    onnx_model = onnx.load(filename)
    actual_opset = onnx_model.opset_import[0].version
    if actual_opset != opset_version:
        print(f"\nNote: Model was exported with opset {actual_opset} (requested {opset_version})")
        print(f"This is normal - PyTorch uses the best compatible opset version.")
        print(f"The version conversion warnings above can be safely ignored.")
        print(f"To avoid warnings, use --opset-version {actual_opset} next time.\n")

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

    # Create quantized version
    filename_int8 = filename.replace(".onnx", ".int8.onnx")
    print(f"Quantizing to INT8...")
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QUInt8,
    )
    print(f"Quantized model saved to: {filename_int8}")


if __name__ == "__main__":
    torch.manual_seed(20251217)
    main()

