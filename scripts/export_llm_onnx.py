#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw

import argparse
import inspect
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qwen import Qwen3UnifiedKvDelta

DEBUG_EXPORT = os.environ.get("DEBUG_EXPORT", "0") == "1"


# Get the first available attribute from a list of attribute names.
def _get_first_attr(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            try:
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = int(v.item())
                    else:
                        continue
                return v
            except Exception:
                continue
    return default


# Create operator set ID for ONNX model.
def _make_opsetid(domain: str, version: int):
    if hasattr(helper, "make_operatorsetid"):
        return helper.make_operatorsetid(domain, version)
    return helper.make_opsetid(domain, version)


# Ensure dual ONNX opset with ai.onnx domain first.
# Reorders opset imports to put ai.onnx before empty domain.
def ensure_dual(model: onnx.ModelProto, target_opset: int):
    others = []
    seen = set()
    for op in model.opset_import:
        if op.domain not in ("", "ai.onnx") and op.domain not in seen:
            others.append(op)
            seen.add(op.domain)
    del model.opset_import[:]
    model.opset_import.extend([
        _make_opsetid("ai.onnx", int(target_opset)),
        _make_opsetid("", int(target_opset)),
    ])
    model.opset_import.extend(others)


# Force IR version and opset to target values.
def force_ir_opset(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    if int(model.ir_version) > int(target_ir_max):
        model.ir_version = int(target_ir_max)
    ensure_dual(model, int(target_opset))


# Remove companion .data file if it exists.
def _strip_companion_data_file(out_path: Path):
    data_path = out_path.with_suffix(".data")
    if data_path.exists():
        try:
            data_path.unlink()
        except Exception:
            pass


# Convert model from external data format to inline if needed.
def _convert_from_external_if_needed(model: onnx.ModelProto):
    try:
        from onnx.external_data_helper import uses_external_data, convert_model_from_external_data
        if uses_external_data(model):
            convert_model_from_external_data(model)
    except Exception:
        pass


# Save ONNX model as a single file (no external data).
def save_onnx_single_file(model: onnx.ModelProto, out_path: str) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    _strip_companion_data_file(out_path)
    _convert_from_external_if_needed(model)
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    _strip_companion_data_file(out_path)
    if out_path.with_suffix(".data").exists():
        raise RuntimeError(f"Unexpected .data generated: {out_path.with_suffix('.data')}")
    print(f"[save] Saved single: {out_path}")
    return str(out_path)


# Save ONNX model with external data in a single .data file.
def save_onnx_external_onefile(model: onnx.ModelProto, out_path: str) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_name = out_path.with_suffix(".data").name
    data_path = out_path.parent / data_name
    if out_path.exists():
        out_path.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )
    print(f"[save] Saved external: {out_path} (+{data_name})")
    return str(out_path)


# Validate ONNX model file using ONNX checker.
def checker_check_file(path: str):
    onnx.checker.check_model(path)


# Set metadata properties in ONNX model.
def set_meta_in_model(model: onnx.ModelProto, meta: Dict[str, Any]) -> None:
    while len(model.metadata_props):
        model.metadata_props.pop()
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key = str(k)
        p.value = str(v)

# Count how many times each value is used in the model graph.
def _all_value_uses(model: onnx.ModelProto) -> Dict[str, int]:
    uses = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                uses[inp] = uses.get(inp, 0) + 1
    for out in model.graph.output:
        if out.name:
            uses[out.name] = uses.get(out.name, 0) + 1
    return uses


# Get constant value from initializer or Constant node.
def _get_const(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    for init in model.graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:
                return None
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    return numpy_helper.to_array(attr.t)
                if attr.name == "value_ints":
                    return np.array(list(attr.ints), dtype=np.int64)
                if attr.name == "value_int":
                    return np.array([attr.i], dtype=np.int64)
    return None


# Remove initializer by name from model graph.
def _remove_initializer(model: onnx.ModelProto, name: str) -> bool:
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            del model.graph.initializer[i]
            return True
    return False


# Remove Constant node that produces the specified output.
def _remove_constant_node_output(model: onnx.ModelProto, output_name: str) -> bool:
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Constant" and node.output and node.output[0] == output_name:
            del model.graph.node[i]
            return True
    return False

# Remove num_outputs attribute from Split nodes (for compatibility).
def fix_split_num_outputs(model: onnx.ModelProto) -> bool:
    modified = False
    for node in model.graph.node:
        if node.op_type == "Split":
            for attr in list(node.attribute):
                if attr.name == "num_outputs":
                    node.attribute.remove(attr)
                    modified = True
    return modified

# Convert Reduce* ops axes input to attribute (for compatibility).
def fix_reduce_axes_input_to_attr(model: onnx.ModelProto) -> bool:
    modified = False
    reduce_ops = {
        "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
        "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
    }
    uses = _all_value_uses(model)
    for node in model.graph.node:
        if node.op_type in reduce_ops and len(node.input) == 2 and node.input[1]:
            axes_name = node.input[1]
            axes_arr = _get_const(model, axes_name)
            if axes_arr is None:
                continue
            axes_list = np.array(axes_arr).astype(np.int64).reshape(-1).tolist()
            node.input.pop()
            for a in list(node.attribute):
                if a.name in ("axes", "noop_with_empty_axes"):
                    node.attribute.remove(a)
            node.attribute.append(helper.make_attribute("axes", axes_list))
            if uses.get(axes_name, 0) <= 1:
                _remove_initializer(model, axes_name)
                _remove_constant_node_output(model, axes_name)
            print(f"[fix] {node.op_type} '{node.name}' axes input -> attr {axes_list}")
            modified = True
    return modified

# Apply all model fixes in place (split, reduce, IR/opset).
def fix_model_inplace(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    fix_split_num_outputs(model)
    fix_reduce_axes_input_to_attr(model)
    force_ir_opset(model, target_ir_max=target_ir_max, target_opset=target_opset)

# Patch ONNX Runtime quantization utils to use target opset version.
def _monkeypatch_ort_get_opset_version(target_opset: int):
    try:
        import onnxruntime.quantization.quant_utils as qutils
        import onnxruntime.quantization.base_quantizer as bq
        try:
            import onnxruntime.quantization.onnx_quantizer as oq
        except Exception:
            oq = None
    except Exception:
        return

    def _patched_get_opset_version(model):
        if isinstance(model, str):
            try:
                model = onnx.load(model, load_external_data=False)
            except Exception:
                return int(target_opset)
        if hasattr(model, "opset_import"):
            ai = None
            empty = None
            for op in model.opset_import:
                if op.domain == "ai.onnx":
                    ai = int(op.version)
                elif op.domain == "":
                    empty = int(op.version)
            if ai is not None:
                return ai
            if empty is not None:
                return empty
        return int(target_opset)

    qutils.get_opset_version = _patched_get_opset_version
    bq.get_opset_version = _patched_get_opset_version
    if oq is not None:
        oq.get_opset_version = _patched_get_opset_version


# Quantize FP32 model to INT8 using dynamic quantization.
# Ensures output is a single file without external data.
def quantize_int8_dynamic_no_data(
    input_fp32_onnx: str,
    output_file: str,
    target_ir_max: int = 9,
    target_opset: int = 17,
):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    _monkeypatch_ort_get_opset_version(int(target_opset))

    sig = inspect.signature(quantize_dynamic)
    kwargs = {}
    if "weight_type" in sig.parameters:
        kwargs["weight_type"] = QuantType.QInt8
    if "per_channel" in sig.parameters:
        kwargs["per_channel"] = True
    if "reduce_range" in sig.parameters:
        kwargs["reduce_range"] = False
    if "extra_options" in sig.parameters:
        kwargs["extra_options"] = {"WeightSymmetric": True}
    if "use_external_data_format" in sig.parameters:
        kwargs["use_external_data_format"] = False

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    _strip_companion_data_file(out_path)

    tmp_dir = out_path.parent / "_tmp_quant"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_quant = tmp_dir / (out_path.stem + ".quant.onnx")
    if tmp_quant.exists():
        tmp_quant.unlink()
    _strip_companion_data_file(tmp_quant)

    print(f"[int8] quantize_dynamic input: {input_fp32_onnx}")
    quantize_dynamic(input_fp32_onnx, str(tmp_quant), **kwargs)

    q = onnx.load(str(tmp_quant), load_external_data=True)
    _convert_from_external_if_needed(q)
    fix_model_inplace(q, target_ir_max=target_ir_max, target_opset=target_opset)
    ensure_dual(q, int(target_opset))
    save_onnx_single_file(q, output_file)
    checker_check_file(output_file)

    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    if Path(output_file).with_suffix(".data").exists():
        raise RuntimeError(f"int8 output produced .data which is not allowed: {Path(output_file).with_suffix('.data')}")

    print(f"[int8] Saved: {output_file}")


# Load LLM state dict from model checkpoint.
# Extracts keys with 'llm.' prefix.
def load_llm_state_dict(model_pt: str) -> Dict[str, torch.Tensor]:
    data = torch.load(model_pt, map_location="cpu")
    state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    llm = {}
    for k, v in state_dict.items():
        if k.startswith("llm."):
            llm[k[4:]] = v
    if not llm:
        raise RuntimeError("No keys with prefix 'llm.' found in model.pt")
    return llm


# Disable sliding window attention in all decoder layers.
def disable_sliding_window(model) -> None:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                if hasattr(attn, "sliding_window"):
                    attn.sliding_window = None
                if hasattr(attn, "config") and hasattr(attn.config, "sliding_window"):
                    attn.config.sliding_window = None
    print("Disabled sliding_window in all decoder layers")


# Build HuggingFace LLM model from config and state dict.
# Sets eager attention implementation and disables sliding window.
def build_hf_llm(config, llm_sd: Dict[str, torch.Tensor], dtype: torch.dtype):
    setattr(config, "_attn_implementation", "eager")
    print("Set config._attn_implementation = 'eager'")
    try:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=dtype)

    missing, unexpected = model.load_state_dict(llm_sd, strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)} (show 5) ->", missing[:5])
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)} (show 5) ->", unexpected[:5])

    disable_sliding_window(model)
    model.eval().to("cpu")
    return model


# Infer attention head configuration from model weights.
# Uses q_proj/k_proj out_features to determine head_dim and num_kv_heads.
def infer_heads_from_model(model) -> Dict[str, int]:
    """
    Qwen3: do NOT derive head_dim from hidden_size/num_heads.
    Use q_proj/k_proj out_features to infer head_dim and kv_heads.
    """
    cfg = model.config
    hidden_size = int(_get_first_attr(cfg, ["hidden_size", "n_embd", "dim"], 0))
    num_heads = int(_get_first_attr(cfg, ["num_attention_heads", "num_heads", "n_head"], 0))
    num_kv_heads_cfg = int(_get_first_attr(cfg, ["num_key_value_heads", "num_kv_heads", "n_head_kv"], 0))

    if not (hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0):
        raise RuntimeError("Model has no model.layers[0] to probe attention projections")

    attn0 = model.model.layers[0].self_attn
    if not (hasattr(attn0, "q_proj") and hasattr(attn0.q_proj, "weight")):
        raise RuntimeError("Cannot access layer0.self_attn.q_proj.weight for head_dim inference")

    q_out = int(attn0.q_proj.weight.shape[0])  # out_features
    if num_heads <= 0:
        # fallback: try attn.num_heads if exists
        num_heads = int(_get_first_attr(attn0, ["num_heads", "n_heads"], 0))
    if num_heads <= 0:
        raise RuntimeError(f"Cannot infer num_heads (config and attn both invalid). q_out={q_out}")

    if q_out % num_heads != 0:
        raise RuntimeError(f"q_proj.out_features not divisible by num_heads: {q_out} % {num_heads} != 0")

    head_dim = int(q_out // num_heads)

    # infer kv heads using k_proj out_features
    if not (hasattr(attn0, "k_proj") and hasattr(attn0.k_proj, "weight")):
        raise RuntimeError("Cannot access layer0.self_attn.k_proj.weight for kv head inference")

    k_out = int(attn0.k_proj.weight.shape[0])
    if k_out % head_dim != 0:
        raise RuntimeError(f"k_proj.out_features not divisible by head_dim: {k_out} % {head_dim} != 0")
    num_kv_heads = int(k_out // head_dim)

    # optional sanity prints
    print(f"heads probe: hidden_size={hidden_size}, q_out={q_out}, k_out={k_out}")
    print(f"heads probe: num_heads={num_heads}, head_dim={head_dim}, num_kv_heads={num_kv_heads} (cfg={num_kv_heads_cfg})")

    return {
        "hidden_size": hidden_size,
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }


# Export PyTorch model to ONNX format.
def _torch_onnx_export(wrapped, inputs, out_path: str, opset: int, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        wrapped,
        inputs,
        out_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,
    )


# Export unified KV delta LLM model to ONNX.
# Creates a model that outputs KV deltas instead of full KV cache.
@torch.no_grad()
def export_unified_llm_kv_delta(
    model,
    hidden_size: int,
    vocab_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    past_len: int,
    max_total_len: int,
    opset: int,
    out_path: str,
):
    wrapped = Qwen3UnifiedKvDelta(model, max_total_len=max_total_len).eval()

    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
    am = torch.ones(1, past_len + seq_len, dtype=torch.int64)
    cache_pos = torch.arange(past_len, past_len + seq_len, dtype=torch.int64)

    cache_dtype = next(model.parameters()).dtype
    cache_flat = []
    for _ in range(num_layers):
        ck = torch.zeros(1, max_total_len, num_kv_heads, head_dim, dtype=cache_dtype)
        cv = torch.zeros(1, max_total_len, num_kv_heads, head_dim, dtype=cache_dtype)
        cache_flat.extend([ck, cv])

    input_names = ["inputs_embeds", "attention_mask", "cache_position"]
    for i in range(num_layers):
        input_names += [f"cache_key_{i}", f"cache_value_{i}"]

    output_names = ["logits"]
    for i in range(num_layers):
        output_names += [f"key_delta_{i}", f"value_delta_{i}"]

    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "total_seq"},
        "cache_position": {0: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(num_layers):
        dynamic_axes[f"cache_key_{i}"] = {0: "batch"}
        dynamic_axes[f"cache_value_{i}"] = {0: "batch"}
        dynamic_axes[f"key_delta_{i}"] = {0: "batch", 1: "seq"}
        dynamic_axes[f"value_delta_{i}"] = {0: "batch", 1: "seq"}

    print(f"Export unified kv-delta llm -> {out_path} (opset={opset}, model_dtype={cache_dtype})")
    _torch_onnx_export(
        wrapped,
        (x, am, cache_pos, *cache_flat),
        out_path,
        opset,
        input_names,
        output_names,
        dynamic_axes,
    )


# Convert ONNX Runtime type string to numpy dtype.
def _np_dtype(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    raise RuntimeError(f"Unsupported ort type: {ort_type}")


# Verify unified KV delta ONNX model with ONNX Runtime.
# Tests both prefill and decode phases.
def verify_unified_kv_delta_onnx(
    llm_path: str,
    hidden_size: int,
    vocab_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_total_len: int,
    prefill_len: int = 16,
):
    import onnxruntime as ort

    print("Verifying unified kv-delta model with onnxruntime (CPUExecutionProvider)...")
    sess = ort.InferenceSession(llm_path, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in sess.get_inputs()}

    x_dtype = _np_dtype(ins["inputs_embeds"].type)
    ck_dtype = _np_dtype(ins["cache_key_0"].type)

    B = 1
    T = int(prefill_len)

    # Prefill
    past_len = 0
    seq_len = T
    total_seq = past_len + seq_len

    x = np.random.randn(B, seq_len, hidden_size).astype(x_dtype)
    am = np.ones((B, total_seq), dtype=np.int64)
    cache_pos = np.arange(past_len, past_len + seq_len, dtype=np.int64)

    cache_k = np.zeros((B, max_total_len, num_kv_heads, head_dim), dtype=ck_dtype)
    cache_v = np.zeros((B, max_total_len, num_kv_heads, head_dim), dtype=ck_dtype)

    feed = {"inputs_embeds": x, "attention_mask": am, "cache_position": cache_pos}
    for i in range(num_layers):
        feed[f"cache_key_{i}"] = cache_k
        feed[f"cache_value_{i}"] = cache_v

    out_prefill = sess.run(None, feed)
    logits = out_prefill[0]
    if not (logits.shape[0] == B and logits.shape[1] == seq_len and logits.shape[-1] == vocab_size):
        raise RuntimeError(f"Prefill logits shape mismatch: {logits.shape}")
    print("Prefill OK. logits:", logits.shape)

    # write deltas into cache [0:T)
    for i in range(num_layers):
        k_delta = out_prefill[1 + 2 * i]        # [B,S,kv,hd]
        v_delta = out_prefill[1 + 2 * i + 1]
        cache_k[:, 0:seq_len, :, :] = k_delta.astype(ck_dtype, copy=False)
        cache_v[:, 0:seq_len, :, :] = v_delta.astype(ck_dtype, copy=False)

    # Decode
    past_len = T
    seq_len = 1
    total_seq = past_len + seq_len

    x = np.random.randn(B, seq_len, hidden_size).astype(x_dtype)
    am = np.ones((B, total_seq), dtype=np.int64)
    cache_pos = np.array([past_len], dtype=np.int64)

    feed = {"inputs_embeds": x, "attention_mask": am, "cache_position": cache_pos}
    for i in range(num_layers):
        feed[f"cache_key_{i}"] = cache_k
        feed[f"cache_value_{i}"] = cache_v

    out_decode = sess.run(None, feed)
    logits2 = out_decode[0]
    if not (logits2.shape[0] == B and logits2.shape[1] == seq_len and logits2.shape[-1] == vocab_size):
        raise RuntimeError(f"Decode logits shape mismatch: {logits2.shape}")
    print("Decode OK. logits:", logits2.shape)

    print("Verify kv-delta OK.")


# Parse command-line arguments for export script.
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pt", type=str, required=True)
    ap.add_argument("--llm-config-path", type=str, required=True)
    ap.add_argument("--output-root", type=str, default="models")
    ap.add_argument("--opset-version", type=int, default=17)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--past-len", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-fp32", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--target-opset", type=int, default=17)
    ap.add_argument("--target-ir-max", type=int, default=9)
    ap.add_argument("--keep-tmp", action="store_true")
    return ap.parse_args()


# Main export function.
# Exports LLM model to ONNX in FP16, FP32, and INT8 formats.
def main():
    args = get_args()
    print(vars(args))

    export_fp32 = not args.no_fp32
    export_fp16 = not args.no_fp16
    export_int8 = not args.no_int8

    llm_sd = load_llm_state_dict(args.model_pt)
    total_params = sum(int(p.numel()) for p in llm_sd.values())
    print(f"LLM param tensors: {len(llm_sd)}, total params: {total_params/1e6:.2f}M ({total_params/1e9:.3f}B)")

    config = AutoConfig.from_pretrained(args.llm_config_path, trust_remote_code=True)
    hidden_size = int(getattr(config, "hidden_size"))
    vocab_size = int(getattr(config, "vocab_size"))
    num_layers = int(getattr(config, "num_hidden_layers", getattr(config, "num_layers", 0)))
    print(f"config: hidden_size={hidden_size}, vocab_size={vocab_size}, num_layers={num_layers}")

    root_dir = Path(args.output_root)
    fp16_dir = root_dir / "llm_fp16"
    fp32_dir = root_dir / "llm_fp32"
    int8_dir = root_dir / "llm_int8"
    tmp_dir = root_dir / "_tmp_export_unified"

    if export_fp16:
        fp16_dir.mkdir(parents=True, exist_ok=True)
    if export_fp32:
        fp32_dir.mkdir(parents=True, exist_ok=True)
    if export_int8:
        int8_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_fp16 = str(tmp_dir / "llm_unified.fp16.raw.onnx")
    raw_fp32 = str(tmp_dir / "llm_unified.fp32.raw.onnx")

    out_fp16 = str(fp16_dir / "llm.fp16.onnx")
    out_fp32 = str(fp32_dir / "llm.fp32.onnx")
    out_int8 = str(int8_dir / "llm.int8.onnx")

    for p in [raw_fp16, raw_fp32, out_fp16, out_fp32, out_int8]:
        pp = Path(p)
        if pp.exists():
            pp.unlink()
        _strip_companion_data_file(pp)

    max_total_len = int(args.seq_len) + int(args.past_len)

    try:
        # heads probe (must be from q_proj/k_proj out_features)
        probe = build_hf_llm(config, llm_sd, dtype=torch.float16)
        heads = infer_heads_from_model(probe)
        num_heads = heads["num_heads"]
        num_kv_heads = heads["num_kv_heads"]
        head_dim = heads["head_dim"]
        print(f"heads: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

        # Base metadata
        meta_base = {
            "model_type": "qwen3_causallm_unified_prefill_decode_kv_delta",
            "version": "1",
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "max_total_len": max_total_len,
            "io_dtype": "float32",
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }

        # FP16 (single-file)
        if export_fp16:
            model_fp16 = build_hf_llm(config, llm_sd, dtype=torch.float16)

            if DEBUG_EXPORT:
                past_len = 16
                seq_len = 16
                x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
                am = torch.ones(1, past_len + seq_len, dtype=torch.int64)
                cache_pos = torch.arange(past_len, past_len + seq_len, dtype=torch.int64)
                cache_dtype = next(model_fp16.parameters()).dtype
                cache_flat = []
                for _ in range(num_layers):
                    ck = torch.zeros(1, max_total_len, num_kv_heads, head_dim, dtype=cache_dtype)
                    cv = torch.zeros(1, max_total_len, num_kv_heads, head_dim, dtype=cache_dtype)
                    cache_flat.extend([ck, cv])
                print(f"[DEBUG_EXPORT] torch dry-run (consistent example) past_len={past_len}, seq_len={seq_len}")
                _ = Qwen3UnifiedKvDelta(model_fp16, max_total_len=max_total_len)(x, am, cache_pos, *cache_flat)
                print("[DEBUG_EXPORT] torch dry-run ok")

            export_unified_llm_kv_delta(
                model_fp16,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                seq_len=int(args.seq_len),
                past_len=int(args.past_len),
                max_total_len=max_total_len,
                opset=int(args.opset_version),
                out_path=raw_fp16,
            )

            m16 = onnx.load(raw_fp16, load_external_data=True)
            fix_model_inplace(m16, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
            ensure_dual(m16, int(args.target_opset))
            meta16 = dict(meta_base)
            meta16["quantization_type"] = "fp16"
            set_meta_in_model(m16, meta16)
            save_onnx_single_file(m16, out_fp16)
            checker_check_file(out_fp16)

        # FP32 (external .data)
        if export_fp32 or export_int8:
            model_fp32 = build_hf_llm(config, llm_sd, dtype=torch.float32)

            if DEBUG_EXPORT:
                past_len = 16
                seq_len = 16
                x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
                am = torch.ones(1, past_len + seq_len, dtype=torch.int64)
                cache_pos = torch.arange(past_len, past_len + seq_len, dtype=torch.int64)
                cache_dtype = next(model_fp32.parameters()).dtype
                cache_flat = []
                for _ in range(num_layers):
                    ck = torch.zeros(1, max_total_len, num_kv_heads, head_dim, dtype=cache_dtype)
                    cv = torch.zeros(1, max_total_len, num_kv_heads, head_dim, dtype=cache_dtype)
                    cache_flat.extend([ck, cv])
                print(f"[DEBUG_EXPORT] torch dry-run (consistent example) past_len={past_len}, seq_len={seq_len}")
                _ = Qwen3UnifiedKvDelta(model_fp32, max_total_len=max_total_len)(x, am, cache_pos, *cache_flat)
                print("[DEBUG_EXPORT] torch dry-run ok")

            export_unified_llm_kv_delta(
                model_fp32,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                seq_len=int(args.seq_len),
                past_len=int(args.past_len),
                max_total_len=max_total_len,
                opset=int(args.opset_version),
                out_path=raw_fp32,
            )

            m32 = onnx.load(raw_fp32, load_external_data=True)
            fix_model_inplace(m32, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
            ensure_dual(m32, int(args.target_opset))
            meta32 = dict(meta_base)
            meta32["quantization_type"] = "fp32"
            set_meta_in_model(m32, meta32)
            save_onnx_external_onefile(m32, out_fp32)
            checker_check_file(out_fp32)

        # INT8 (single-file)
        if export_int8:
            if not export_fp32:
                raise RuntimeError("int8 export requires fp32 model (as quantization source). Please enable fp32 export.")

            quantize_int8_dynamic_no_data(out_fp32, out_int8, target_ir_max=args.target_ir_max, target_opset=args.target_opset)

            mi8 = onnx.load(out_int8, load_external_data=True)
            fix_model_inplace(mi8, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
            ensure_dual(mi8, int(args.target_opset))
            meta8 = dict(meta_base)
            meta8["quantization_type"] = "int8"
            set_meta_in_model(mi8, meta8)
            save_onnx_single_file(mi8, out_int8)
            checker_check_file(out_int8)

        # Verify (CPU ORT)
        if args.verify:
            if export_fp16:
                verify_unified_kv_delta_onnx(
                    out_fp16, hidden_size, vocab_size, num_layers, num_kv_heads, head_dim, max_total_len, prefill_len=16
                )
            if export_fp32:
                verify_unified_kv_delta_onnx(
                    out_fp32, hidden_size, vocab_size, num_layers, num_kv_heads, head_dim, max_total_len, prefill_len=16
                )
            if export_int8:
                verify_unified_kv_delta_onnx(
                    out_int8, hidden_size, vocab_size, num_layers, num_kv_heads, head_dim, max_total_len, prefill_len=16
                )

    finally:
        # Delete tmp no matter success/fail (unless keep_tmp)
        if not args.keep_tmp:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            # also delete quant tmp if any
            try:
                shutil.rmtree(root_dir / "llm_int8" / "_tmp_quant", ignore_errors=True)
            except Exception:
                pass

    print("Done.")
    if export_fp16:
        print(" ", out_fp16)
    if export_fp32:
        print(" ", out_fp32, "+", str(Path(out_fp32).with_suffix(".data")))
    if export_int8:
        print(" ", out_int8)


if __name__ == "__main__":
    torch.manual_seed(20260104)
    main()
