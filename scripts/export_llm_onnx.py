#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCausalLM, AutoConfig

try:
    from transformers import masking_utils
except Exception:
    masking_utils = None

# Patch Qwen3's create_causal_mask to be ONNX-export friendly
# The original implementation may use operations that are not well-supported in ONNX
if masking_utils is not None:

    def create_causal_mask(
        *,
        input_ids_shape=None,
        input_shape=None,
        attention_mask=None,
        dtype=None,
        device=None,
        past_key_values_length=0,
        sliding_window=None,
        cache_position=None,
        **kwargs,
    ):
        shape = input_ids_shape if input_ids_shape is not None else input_shape
        if shape is None and attention_mask is not None:
            shape = attention_mask.shape
        if shape is None or len(shape) < 2:
            raise ValueError("create_causal_mask: cannot infer shape")
        batch_size, tgt_len = shape[0], shape[1]
        if isinstance(batch_size, torch.Tensor):
            batch_size = int(batch_size)
        if isinstance(tgt_len, torch.Tensor):
            tgt_len = int(tgt_len)
        if dtype is None:
            dtype = torch.float32
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            device = attention_mask.device
        elif device is None:
            device = torch.device("cpu")
        if isinstance(past_key_values_length, torch.Tensor):
            past = int(past_key_values_length.item())
        else:
            past = int(past_key_values_length)
        src_len = tgt_len + past
        min_value = torch.finfo(dtype).min
        mask = torch.full(
            (batch_size, 1, tgt_len, src_len),
            fill_value=min_value,
            dtype=dtype,
            device=device,
        )
        q_pos = torch.arange(tgt_len, device=device).unsqueeze(-1)
        k_pos = torch.arange(src_len, device=device).unsqueeze(0)
        cond = k_pos <= (past + q_pos)
        cond = cond.view(1, 1, tgt_len, src_len)
        zero = torch.zeros(1, dtype=dtype, device=device)
        mask = torch.where(cond, zero, mask)
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask, device=device)
            if attention_mask.dim() == 2:
                if attention_mask.shape[1] < src_len:
                    pad = src_len - attention_mask.shape[1]
                    attention_mask = F.pad(attention_mask, (0, pad), value=0)
                elif attention_mask.shape[1] > src_len:
                    attention_mask = attention_mask[:, :src_len]
                attn = (attention_mask > 0).to(dtype)
                pad_mask = (1.0 - attn) * min_value
                pad_mask = pad_mask.view(batch_size, 1, 1, src_len)
                mask = mask + pad_mask
        return mask

    # Replace the original create_causal_mask with our export-friendly version
    masking_utils.create_causal_mask = create_causal_mask
    print("Patched Qwen3 create_causal_mask with export-friendly causal mask")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-pt",
        type=str,
        required=True,
        help="Path to model.pt file containing LLM parameters",
    )
    parser.add_argument(
        "--llm-config-path",
        type=str,
        required=True,
        help="Path to LLM config directory (e.g., Qwen3-0.6B)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="llm.onnx",
        help="Output ONNX filename (fp16 model)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Dummy sequence length for export",
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


# Wrapper to simplify ONNX export: only output logits, disable KV cache
class LLMWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inputs_embeds, attention_mask):
        out = self.m(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,  # Disable KV cache for simpler ONNX graph
        )
        return out.logits  # Only return logits for token prediction


@torch.no_grad()
def export_onnx(
    model: torch.nn.Module,
    config,
    hidden_size: int,
    filename: str,
    seq_length: int,
    opset_version: int,
    dtype: torch.dtype,
):
    """
    Export PyTorch model to ONNX format with dynamic sequence length support.
    The exported model can handle variable-length sequences at inference time.
    """
    wrapped = LLMWrapper(model).eval()
    batch_size = 1
    # Create dummy inputs for ONNX export (actual sequence length will be dynamic)
    inputs_embeds = torch.randn(
        batch_size,
        seq_length,
        hidden_size,
        dtype=dtype,
    )
    attention_mask = torch.ones(
        batch_size,
        seq_length,
        dtype=torch.int64,
    )
    print(f"Creating dummy inputs: inputs_embeds shape={inputs_embeds.shape}")
    print("Note: ONNX model will use dynamic sequence_length")
    os.environ["TORCH_ONNX_DISABLE_DYNAMO"] = "1"  # Disable TorchDynamo for compatibility
    print(f"Exporting to ONNX (opset {opset_version}) -> {filename}")
    torch.onnx.export(
        wrapped,
        (inputs_embeds, attention_mask),
        filename,
        opset_version=opset_version,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits"],
        # Enable dynamic axes for batch_size and sequence_length
        dynamic_axes={
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        do_constant_folding=True,  # Optimize constant operations
        verbose=False,
        export_params=True,  # Export model parameters
    )
    print(f"ONNX model saved to: {filename}")


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    pt_path = Path(args.model_pt)
    if not pt_path.exists():
        raise ValueError(f"model.pt not found: {pt_path}")

    print(f"Loading model.pt from {pt_path}...")
    data = torch.load(pt_path, map_location="cpu")
    state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data

    # Extract LLM parameters from model.pt
    # Remove "llm." prefix from keys to match HuggingFace model structure
    llm_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("llm."):
            new_key = k[4:]  # Remove "llm." prefix
            llm_state_dict[new_key] = v
    if len(llm_state_dict) == 0:
        print("Error: no keys start with 'llm.' in state_dict")
        print("First 20 keys:")
        for k in list(state_dict.keys())[:20]:
            print(" ", k)
        raise ValueError("Failed to extract LLM weights")

    print(f"Found {len(llm_state_dict)} LLM parameter groups (state_dict keys)")
    total_params = sum(p.numel() for p in llm_state_dict.values())
    total_params_m = total_params / 1e6
    print(f"Total LLM parameters: {total_params_m:.2f}M ({total_params / 1e9:.3f}B)")

    print(f"Loading LLM config from {args.llm_config_path}...")
    config = AutoConfig.from_pretrained(args.llm_config_path, trust_remote_code=True)
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    print(f"Model config: hidden_size={hidden_size}, vocab_size={vocab_size}")

    # Use eager attention implementation for ONNX export compatibility
    # Flash attention and other optimized implementations may not export correctly
    config._attn_implementation = "eager"
    print("Set config._attn_implementation = 'eager' for export")

    # Export fp16 model first (original precision, matches model.pt)
    # This preserves the original model's precision and is suitable for GPU inference
    print("Creating fp16 LLM model structure...")
    model_fp16 = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print("Loading LLM parameters into fp16 model...")
    missing_keys, unexpected_keys = model_fp16.load_state_dict(llm_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys (first 10):")
        for k in missing_keys[:10]:
            print("  ", k)
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys (first 10):")
        for k in unexpected_keys[:10]:
            print("  ", k)

    # Disable sliding window attention for ONNX export compatibility
    # Sliding window may cause issues during ONNX conversion
    if hasattr(model_fp16, "model") and hasattr(model_fp16.model, "layers"):
        for layer in model_fp16.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                if hasattr(attn, "sliding_window"):
                    attn.sliding_window = None
                if hasattr(attn, "config"):
                    setattr(attn.config, "sliding_window", None)
        print("Force disabled sliding_window in all decoder layers (fp16)")

    model_fp16.eval()
    model_fp16.to("cpu")

    seq_length = int(args.seq_length)
    opset_version = int(args.opset_version)
    filename = args.output_filename

    export_onnx(
        model_fp16,
        config,
        hidden_size,
        filename,
        seq_length,
        opset_version,
        torch.float16,
    )

    print("Verifying exported fp16 model with ONNX Runtime...")
    import onnxruntime as ort

    sess = ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
    print("Inputs:")
    for inp in sess.get_inputs():
        print(f"  - {inp.name}: {inp.shape}, {inp.type}")
    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  - {out.name}: {out.shape}, {out.type}")

    test_len = seq_length
    import numpy as np

    test_inputs_embeds = np.random.randn(1, test_len, hidden_size).astype("float16")
    test_attention_mask = np.ones((1, test_len), dtype="int64")
    out = sess.run(
        None,
        {
            "inputs_embeds": test_inputs_embeds,
            "attention_mask": test_attention_mask,
        },
    )
    print(f"ONNX forward test ok (length={test_len}), logits shape: {out[0].shape}")

    test_len_short = 80
    test_inputs_embeds_short = np.random.randn(1, test_len_short, hidden_size).astype(
        "float16"
    )
    test_attention_mask_short = np.ones((1, test_len_short), dtype="int64")
    out_short = sess.run(
        None,
        {
            "inputs_embeds": test_inputs_embeds_short,
            "attention_mask": test_attention_mask_short,
        },
    )
    print(f"ONNX forward test ok (length={test_len_short}), logits shape: {out_short[0].shape}")

    model_author = "FunAudioLLM"
    comment = os.environ.get("comment", "FunAudioLLM/Fun-ASR-Nano-2512")
    url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512"
    meta_data = {
        "model_type": "qwen3_causallm",
        "version": "1",
        "model_author": model_author,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "comment": comment,
        "url": url,
    }
    add_meta_data(filename, meta_data)
    print("Meta data added to fp16 model.")

    filename_int8 = filename.replace(".onnx", ".int8.onnx")
    tmp_fp32 = filename.replace(".onnx", ".fp32.onnx")

    # Export fp32 model for INT8 quantization
    # quantize_dynamic requires float32 input model, cannot quantize float16 directly
    print("Creating fp32 LLM model for INT8 quantization...")
    model_fp32 = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    print("Loading LLM parameters into fp32 model...")
    model_fp32.load_state_dict(llm_state_dict, strict=False)
    # Disable sliding window for fp32 model as well
    if hasattr(model_fp32, "model") and hasattr(model_fp32.model, "layers"):
        for layer in model_fp32.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                if hasattr(attn, "sliding_window"):
                    attn.sliding_window = None
                if hasattr(attn, "config"):
                    setattr(attn.config, "sliding_window", None)
        print("Force disabled sliding_window in all decoder layers (fp32)")
    model_fp32.eval()
    model_fp32.to("cpu")

    print(f"Exporting fp32 ONNX to {tmp_fp32} for quantization...")
    export_onnx(
        model_fp32,
        config,
        hidden_size,
        tmp_fp32,
        seq_length,
        opset_version,
        torch.float32,
    )

    # Quantize MatMul operations to INT8 (weights only, activations remain float32)
    # This reduces model size and improves inference speed with minimal accuracy loss
    print("Quantizing fp32 ONNX to INT8 (weights only)...")
    quantize_dynamic(
        model_input=tmp_fp32,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],  # Only quantize matrix multiplication weights
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,  # Required for large models
    )
    add_meta_data(filename_int8, meta_data)
    print(f"Quantized model saved to: {filename_int8}")

    # Clean up temporary fp32 model files after quantization
    try:
        os.remove(tmp_fp32)
        data_file = tmp_fp32 + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
    except Exception:
        pass


if __name__ == "__main__":
    torch.manual_seed(20251218)
    main()
