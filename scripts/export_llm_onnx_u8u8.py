#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
#
# This version is designed to work around a bug/incompatibility in ONNX Runtime's INT8 kernel
# on certain CPUs (especially AVX2-only paths), where quantized operator outputs become
# completely distorted. This version provides multiple INT8 quantization variants:
#   - u8s8_rr:     U8S8 with per_channel=True,  reduce_range=True  (AVX2-safe)
#   - u8s8_rr_pt:  U8S8 with per_channel=False, reduce_range=True  (AVX2-safe, per-tensor)
#   - u8u8:        U8U8 with per_channel=True,  reduce_range=False (most robust)
#
# The default output (llm.int8.onnx) uses u8u8 mode when exporting all variants,
# as it provides the best compatibility across different CPU architectures.

import argparse
import inspect
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qwen import Qwen3UnifiedKvDelta

DEBUG_EXPORT = os.environ.get("DEBUG_EXPORT", "0") == "1"



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


def _make_opsetid(domain: str, version: int):
    if hasattr(helper, "make_operatorsetid"):
        return helper.make_operatorsetid(domain, version)
    return helper.make_opsetid(domain, version)


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


def force_ir_opset(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    if int(model.ir_version) > int(target_ir_max):
        model.ir_version = int(target_ir_max)
    ensure_dual(model, int(target_opset))


def _strip_companion_data_file(out_path: Path):
    data_path = out_path.with_suffix(".data")
    if data_path.exists():
        try:
            data_path.unlink()
        except Exception:
            pass

def _convert_from_external_if_needed(model: onnx.ModelProto):
    try:
        from onnx.external_data_helper import uses_external_data, convert_model_from_external_data
        if uses_external_data(model):
            convert_model_from_external_data(model)
    except Exception:
        pass

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


def checker_check_file(path: str):
    onnx.checker.check_model(path)


def set_meta_in_model(model: onnx.ModelProto, meta: Dict[str, Any]) -> None:
    while len(model.metadata_props):
        model.metadata_props.pop()
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key = str(k)
        p.value = str(v)


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


def _remove_initializer(model: onnx.ModelProto, name: str) -> bool:
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            del model.graph.initializer[i]
            return True
    return False


def _remove_constant_node_output(model: onnx.ModelProto, output_name: str) -> bool:
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Constant" and node.output and node.output[0] == output_name:
            del model.graph.node[i]
            return True
    return False


def fix_split_num_outputs(model: onnx.ModelProto) -> bool:
    modified = False
    for node in model.graph.node:
        if node.op_type == "Split":
            for attr in list(node.attribute):
                if attr.name == "num_outputs":
                    node.attribute.remove(attr)
                    modified = True
    return modified


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


def fix_model_inplace(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    fix_split_num_outputs(model)
    fix_reduce_axes_input_to_attr(model)
    force_ir_opset(model, target_ir_max=target_ir_max, target_opset=target_opset)


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


# This function supports both QInt8 (U8S8) and QUInt8 (U8U8) weight types,
# while the original version only supports QInt8. This allows generating multiple
# INT8 variants to work around ORT INT8 kernel compatibility issues on AVX2-only CPUs.
def quantize_8bit_dynamic_no_data(
    input_fp32_onnx: str,
    output_file: str,
    weight_type: str = "qint8",          # "qint8" => U8S8, "quint8" => U8U8
    per_channel: bool = True,
    reduce_range: bool = True,
    weight_symmetric: Optional[bool] = None,
    target_ir_max: int = 9,
    target_opset: int = 17,
):

    from onnxruntime.quantization import QuantType, quantize_dynamic

    _monkeypatch_ort_get_opset_version(int(target_opset))

    wt = QuantType.QInt8 if weight_type.lower() == "qint8" else QuantType.QUInt8

    sig = inspect.signature(quantize_dynamic)
    kwargs = {}

    if "weight_type" in sig.parameters:
        kwargs["weight_type"] = wt
    if "per_channel" in sig.parameters:
        kwargs["per_channel"] = bool(per_channel)
    if "reduce_range" in sig.parameters:
        kwargs["reduce_range"] = bool(reduce_range)
    if "use_external_data_format" in sig.parameters:
        kwargs["use_external_data_format"] = False

    if "extra_options" in sig.parameters:
        # DIFFERENCE: Auto-determine weight_symmetric based on weight_type
        # QInt8 uses symmetric quantization, QUInt8 uses asymmetric
        # This is important for AVX2 compatibility
        if weight_symmetric is None:
            weight_symmetric = (wt == QuantType.QInt8)
        kwargs["extra_options"] = {"WeightSymmetric": bool(weight_symmetric)}

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

    print(f"[8bit] quantize_dynamic input: {input_fp32_onnx}")
    print(f"[8bit] -> {output_file}")
    print(f"[8bit] args: weight_type={weight_type}, per_channel={per_channel}, reduce_range={reduce_range}, "
          f"WeightSymmetric={kwargs.get('extra_options', {}).get('WeightSymmetric', None)}")

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
        raise RuntimeError(f"8bit output produced .data which is not allowed: {Path(output_file).with_suffix('.data')}")

    print(f"[8bit] Saved: {output_file}")



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


def infer_heads_from_model(model) -> Dict[str, int]:
    cfg = model.config
    hidden_size = int(_get_first_attr(cfg, ["hidden_size", "n_embd", "dim"], 0))
    num_heads = int(_get_first_attr(cfg, ["num_attention_heads", "num_heads", "n_head"], 0))
    num_kv_heads_cfg = int(_get_first_attr(cfg, ["num_key_value_heads", "num_kv_heads", "n_head_kv"], 0))

    if not (hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0):
        raise RuntimeError("Model has no model.layers[0] to probe attention projections")

    attn0 = model.model.layers[0].self_attn
    if not (hasattr(attn0, "q_proj") and hasattr(attn0.q_proj, "weight")):
        raise RuntimeError("Cannot access layer0.self_attn.q_proj.weight for head_dim inference")

    q_out = int(attn0.q_proj.weight.shape[0])
    if num_heads <= 0:
        num_heads = int(_get_first_attr(attn0, ["num_heads", "n_heads"], 0))
    if num_heads <= 0:
        raise RuntimeError(f"Cannot infer num_heads (config and attn both invalid). q_out={q_out}")

    if q_out % num_heads != 0:
        raise RuntimeError(f"q_proj.out_features not divisible by num_heads: {q_out} % {num_heads} != 0")

    head_dim = int(q_out // num_heads)

    if not (hasattr(attn0, "k_proj") and hasattr(attn0.k_proj, "weight")):
        raise RuntimeError("Cannot access layer0.self_attn.k_proj.weight for kv head inference")

    k_out = int(attn0.k_proj.weight.shape[0])
    if k_out % head_dim != 0:
        raise RuntimeError(f"k_proj.out_features not divisible by head_dim: {k_out} % {head_dim} != 0")
    num_kv_heads = int(k_out // head_dim)

    print(f"heads probe: hidden_size={hidden_size}, q_out={q_out}, k_out={k_out}")
    print(f"heads probe: num_heads={num_heads}, head_dim={head_dim}, num_kv_heads={num_kv_heads} (cfg={num_kv_heads_cfg})")

    return {
        "hidden_size": hidden_size,
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }


def _torch_onnx_export(wrapped, inputs, out_path: str, opset: int, input_names, output_names, dynamic_axes):
    try:
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
            dynamo=True,
        )
        return
    except Exception as e:
        print("[WARN] dynamo export failed, fallback to legacy exporter:", repr(e))

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
        "logits": {0: "batch"},
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


def _np_dtype(ort_type: str):
    s = str(ort_type).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    raise RuntimeError(f"Unsupported ort type: {ort_type}")


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
    if not (logits.shape[0] == B and logits.shape[1] == 1 and logits.shape[-1] == vocab_size):
        raise RuntimeError(f"Prefill logits shape mismatch: {logits.shape}")
    print("Prefill OK. logits:", logits.shape)

    for i in range(num_layers):
        k_delta = out_prefill[1 + 2 * i]
        v_delta = out_prefill[1 + 2 * i + 1]
        cache_k[:, 0:seq_len, :, :] = k_delta.astype(ck_dtype, copy=False)
        cache_v[:, 0:seq_len, :, :] = v_delta.astype(ck_dtype, copy=False)

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
    if not (logits2.shape[0] == B and logits2.shape[1] == 1 and logits2.shape[-1] == vocab_size):
        raise RuntimeError(f"Decode logits shape mismatch: {logits2.shape}")
    print("Decode OK. logits:", logits2.shape)

    print("Verify kv-delta OK.")


# This version adds --int8-mode parameter to select different INT8 variants
# for AVX2 compatibility. The original version only exports a single INT8 model.
# Removed --no-fp32, --no-fp16, --no-int8 flags (this version focuses on INT8 only).
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pt", type=str, required=True)
    ap.add_argument("--llm-config-path", type=str, required=True)
    ap.add_argument("--output-root", type=str, default="models")
    ap.add_argument("--opset-version", type=int, default=17)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--past-len", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--target-opset", type=int, default=17)
    ap.add_argument("--target-ir-max", type=int, default=9)
    ap.add_argument("--keep-tmp", action="store_true")

    # New --int8-mode parameter to select robust INT8 variants for AVX2-only path
    # These variants work around ORT INT8 kernel bugs that cause distorted outputs
    ap.add_argument(
        "--int8-mode",
        type=str,
        default="all",
        choices=["u8s8_rr", "u8s8_rr_pt", "u8u8", "all"],
        help=(
            "8bit export mode(s):\n"
            "  u8s8_rr     : weight QInt8 (U8S8), per_channel=True,  reduce_range=True\n"
            "  u8s8_rr_pt  : weight QInt8 (U8S8), per_channel=False, reduce_range=True\n"
            "  u8u8        : weight QUInt8 (U8U8), per_channel=True,  reduce_range=False\n"
            "  all         : export all 3 above; also write llm.int8.onnx = u8u8 by default"
        ),
    )
    return ap.parse_args()


def main():
    args = get_args()
    print(vars(args))

    llm_sd = load_llm_state_dict(args.model_pt)
    total_params = sum(int(p.numel()) for p in llm_sd.values())
    print(f"LLM param tensors: {len(llm_sd)}, total params: {total_params/1e6:.2f}M ({total_params/1e9:.3f}B)")

    config = AutoConfig.from_pretrained(args.llm_config_path, trust_remote_code=True)
    hidden_size = int(getattr(config, "hidden_size"))
    vocab_size = int(getattr(config, "vocab_size"))
    num_layers = int(getattr(config, "num_hidden_layers", getattr(config, "num_layers", 0)))
    print(f"config: hidden_size={hidden_size}, vocab_size={vocab_size}, num_layers={num_layers}")

    root_dir = Path(args.output_root)
    int8_dir = root_dir / "llm_int8_compat"

    # tmp dir: fp32 quant source lives here only
    tmp_dir = root_dir / "_tmp_export_unified_fast"
    int8_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # temp raw export + temp fp32 quant source (external .data)
    raw_fp32 = str(tmp_dir / "llm_unified.fp32.raw.onnx")
    tmp_fp32 = str(tmp_dir / "llm.fp32.quant_source.onnx")  # + .data

    out_int8_default = str(int8_dir / "llm.int8.onnx")

    # robust candidates:
    out_u8s8_rr = str(int8_dir / "llm.int8.u8s8.rr.onnx")
    out_u8s8_rr_pt = str(int8_dir / "llm.int8.u8s8.rr.pt.onnx")
    out_u8u8 = str(int8_dir / "llm.uint8.u8u8.onnx")

    # cleanup targets
    for p in [raw_fp32, tmp_fp32, out_int8_default, out_u8s8_rr, out_u8s8_rr_pt, out_u8u8]:
        pp = Path(p)
        if pp.exists():
            try:
                pp.unlink()
            except Exception:
                pass
        _strip_companion_data_file(pp)

    max_total_len = int(args.seq_len) + int(args.past_len)

    try:
        # heads probe
        probe = build_hf_llm(config, llm_sd, dtype=torch.float16)
        heads = infer_heads_from_model(probe)
        num_heads = heads["num_heads"]
        num_kv_heads = heads["num_kv_heads"]
        head_dim = heads["head_dim"]
        print(f"heads: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

        meta_base = {
            "model_type": "qwen3_causallm_unified_prefill_decode_kv_delta",
            "version": "2",
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "max_total_len": max_total_len,
            "io_dtype": "float32",
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "logits_mode": "last",  # [B,1,V]
        }

        # Build fp32 model in memory and export temp fp32 onnx as quantization source
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
            print(f"[DEBUG_EXPORT] torch dry-run past_len={past_len}, seq_len={seq_len}")
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

        # metadata for tmp fp32 quant source (not exported)
        meta32 = dict(meta_base)
        meta32["quantization_type"] = "fp32"
        set_meta_in_model(m32, meta32)

        # Save temp fp32 as external onefile to avoid >2GB inline issues
        save_onnx_external_onefile(m32, tmp_fp32)
        checker_check_file(tmp_fp32)

        # 8bit (single-file) - robust modes
        modes = [args.int8_mode]
        if args.int8_mode == "all":
            modes = ["u8s8_rr", "u8s8_rr_pt", "u8u8"]

        exported = []

        for mode in modes:
            if mode == "u8s8_rr":
                out_file = out_u8s8_rr
                quantize_8bit_dynamic_no_data(
                    tmp_fp32,
                    out_file,
                    weight_type="qint8",
                    per_channel=True,
                    reduce_range=True,
                    weight_symmetric=True,
                    target_ir_max=args.target_ir_max,
                    target_opset=args.target_opset,
                )
                exported.append(("u8s8_rr", out_file, "QInt8", True, True))

            elif mode == "u8s8_rr_pt":
                out_file = out_u8s8_rr_pt
                quantize_8bit_dynamic_no_data(
                    tmp_fp32,
                    out_file,
                    weight_type="qint8",
                    per_channel=False,
                    reduce_range=True,
                    weight_symmetric=True,
                    target_ir_max=args.target_ir_max,
                    target_opset=args.target_opset,
                )
                exported.append(("u8s8_rr_pt", out_file, "QInt8", False, True))

            elif mode == "u8u8":
                out_file = out_u8u8
                quantize_8bit_dynamic_no_data(
                    tmp_fp32,
                    out_file,
                    weight_type="quint8",
                    per_channel=True,
                    reduce_range=False,
                    weight_symmetric=False,
                    target_ir_max=args.target_ir_max,
                    target_opset=args.target_opset,
                )
                exported.append(("u8u8", out_file, "QUInt8", True, False))

            else:
                raise RuntimeError(f"Unknown int8 mode: {mode}")

            # Patch metadata after quantization with mode-specific information
            # Original version sets simpler metadata
            m8 = onnx.load(out_file, load_external_data=True)
            fix_model_inplace(m8, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
            ensure_dual(m8, int(args.target_opset))
            meta8 = dict(meta_base)
            meta8["quantization_type"] = "int8"
            meta8["int8_mode"] = mode  # Records which variant this is

            _mode, _path, _wt, _pc, _rr = exported[-1]
            meta8["weight_type"] = _wt      # Records weight type (QInt8/QUInt8)
            meta8["per_channel"] = int(_pc) # Records per_channel setting
            meta8["reduce_range"] = int(_rr) # Records reduce_range setting

            set_meta_in_model(m8, meta8)
            save_onnx_single_file(m8, out_file)
            checker_check_file(out_file)

        # Create default llm.int8.onnx by copying the most compatible variant
        # When exporting all variants, default to u8u8 (most robust)
        # This allows users to use llm.int8.onnx without knowing which variant works
        if args.int8_mode == "all":
            shutil.copyfile(out_u8u8, out_int8_default)
            print(f"[8bit] default llm.int8.onnx <- {Path(out_u8u8).name}")
        else:
            if args.int8_mode == "u8s8_rr":
                shutil.copyfile(out_u8s8_rr, out_int8_default)
            elif args.int8_mode == "u8s8_rr_pt":
                shutil.copyfile(out_u8s8_rr_pt, out_int8_default)
            elif args.int8_mode == "u8u8":
                shutil.copyfile(out_u8u8, out_int8_default)
            print(f"[8bit] default llm.int8.onnx <- mode={args.int8_mode}")

        checker_check_file(out_int8_default)

        if args.verify:
            verify_unified_kv_delta_onnx(
                out_int8_default,
                hidden_size,
                vocab_size,
                num_layers,
                num_kv_heads,
                head_dim,
                max_total_len,
                prefill_len=16,
            )

    finally:
        if not args.keep_tmp:
            # remove tmp export directory (fp32 artifacts live here)
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            # remove quant temp
            try:
                shutil.rmtree(int8_dir / "_tmp_quant", ignore_errors=True)
            except Exception:
                pass

    print("Done.")
    print(" ", out_int8_default)
    print(" ", out_u8s8_rr)
    print(" ", out_u8s8_rr_pt)
    print(" ", out_u8u8)


if __name__ == "__main__":
    torch.manual_seed(20260123)
    main()
