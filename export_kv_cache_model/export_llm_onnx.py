#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import masking_utils
except Exception:
    masking_utils = None

DEBUG_CACHE = os.environ.get("DEBUG_CACHE", "0") == "1"


def _shape_as_tensor(x: torch.Tensor) -> torch.Tensor:
    try:
        from torch.onnx import operators
        return operators.shape_as_tensor(x)
    except Exception:
        return torch.tensor(list(x.shape), device=x.device, dtype=torch.long)


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
        if dtype is None:
            dtype = torch.float32

        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            device = attention_mask.device
        elif device is None:
            device = torch.device("cpu")

        if cache_position is not None and isinstance(cache_position, torch.Tensor):
            tgt_len_t = _shape_as_tensor(cache_position)[0]
        else:
            shape = input_ids_shape if input_ids_shape is not None else input_shape
            if shape is None and isinstance(attention_mask, torch.Tensor):
                shape = attention_mask.shape
            if shape is None or len(shape) < 2:
                raise ValueError("create_causal_mask: cannot infer shape")
            tgt_len_t = torch.as_tensor(int(shape[1]), device=device, dtype=torch.long)

        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            am_shape = _shape_as_tensor(attention_mask)
            src_len_t = am_shape[1]
            past_t = src_len_t - tgt_len_t
        else:
            past_t = torch.as_tensor(past_key_values_length, device=device, dtype=torch.long)
            src_len_t = tgt_len_t + past_t

        q_pos = torch.arange(tgt_len_t, device=device, dtype=torch.long).unsqueeze(1)
        k_pos = torch.arange(src_len_t, device=device, dtype=torch.long).unsqueeze(0)
        cond = k_pos <= (past_t + q_pos)

        min_value = torch.finfo(dtype).min
        zero = torch.zeros((), dtype=dtype, device=device)
        neg_inf = torch.full((), min_value, dtype=dtype, device=device)
        mask = torch.where(cond.unsqueeze(0).unsqueeze(0), zero, neg_inf)

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask, device=device)
            if attention_mask.dim() == 2:
                attn = (attention_mask > 0).to(dtype)
                pad_mask = (1.0 - attn) * min_value
                pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
                mask = mask + pad_mask

        return mask

    masking_utils.create_causal_mask = create_causal_mask
    print("Patched transformers.masking_utils.create_causal_mask (ONNX friendly)")


def _make_opsetid(domain: str, version: int):
    if hasattr(helper, "make_operatorsetid"):
        return helper.make_operatorsetid(domain, version)
    return helper.make_opsetid(domain, version)


def _ops_to_str(m: onnx.ModelProto):
    return [f"{op.domain}:{op.version}" for op in m.opset_import]


def ensure_dual_onnx_opset_ai_first(model: onnx.ModelProto, target_opset: int):
    others = []
    seen = set()
    for op in model.opset_import:
        if op.domain not in ("", "ai.onnx") and op.domain not in seen:
            others.append(op)
            seen.add(op.domain)
    del model.opset_import[:]
    model.opset_import.extend([_make_opsetid("ai.onnx", int(target_opset)), _make_opsetid("", int(target_opset))])
    model.opset_import.extend(others)


def force_ir_opset(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    if int(model.ir_version) > int(target_ir_max):
        model.ir_version = int(target_ir_max)
    ensure_dual_onnx_opset_ai_first(model, int(target_opset))


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
    return str(out_path)


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
    return uses


def _get_const(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    return numpy_helper.to_array(attr.t)
                if attr.name == "value_ints":
                    return np.array(list(attr.ints), dtype=np.int64)
                if attr.name == "value_int":
                    return np.array([attr.i], dtype=np.int64)
                if attr.name == "value_floats":
                    return np.array(list(attr.floats), dtype=np.float32)
                if attr.name == "value_float":
                    return np.array([attr.f], dtype=np.float32)
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


def convert_fp16_model_to_fp32_inplace(model: onnx.ModelProto):
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                    t = a.t
                    if t.data_type == TensorProto.FLOAT16:
                        arr = numpy_helper.to_array(t).astype(np.float32)
                        a.t.CopyFrom(numpy_helper.from_array(arr, name=t.name if t.name else ""))
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and int(a.i) == int(TensorProto.FLOAT16):
                    a.i = int(TensorProto.FLOAT)

    def _fix_vi(vi):
        tt = vi.type.tensor_type
        if tt and tt.elem_type == TensorProto.FLOAT16:
            tt.elem_type = TensorProto.FLOAT

    for vi in model.graph.input:
        _fix_vi(vi)
    for vi in model.graph.output:
        _fix_vi(vi)
    for vi in model.graph.value_info:
        _fix_vi(vi)


def fix_model_inplace(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    fix_split_num_outputs(model)
    fix_reduce_axes_input_to_attr(model)
    force_ir_opset(model, target_ir_max=target_ir_max, target_opset=target_opset)


def fix_save_fp16_single(input_file: str, output_file: str, meta: Dict[str, Any], target_ir_max: int = 9, target_opset: int = 17):
    m = onnx.load(input_file, load_external_data=True)
    print(f"[fix-fp16] Loading: {input_file}")
    print(f"[fix-fp16] Before IR={m.ir_version}, opset={_ops_to_str(m)}")
    fix_model_inplace(m, target_ir_max=target_ir_max, target_opset=target_opset)
    ensure_dual_onnx_opset_ai_first(m, int(target_opset))
    set_meta_in_model(m, meta)
    print(f"[fix-fp16] After  IR={m.ir_version}, opset={_ops_to_str(m)}")
    save_onnx_single_file(m, output_file)
    onnx.checker.check_model(output_file)
    print(f"[fix-fp16] Saved: {output_file}")


def fix_save_fp32_external(input_file: str, output_file: str, meta: Dict[str, Any], target_ir_max: int = 9, target_opset: int = 17):
    m = onnx.load(input_file, load_external_data=True)
    print(f"[fix-fp32] Loading: {input_file}")
    print(f"[fix-fp32] Before IR={m.ir_version}, opset={_ops_to_str(m)}")
    fix_model_inplace(m, target_ir_max=target_ir_max, target_opset=target_opset)
    convert_fp16_model_to_fp32_inplace(m)
    force_ir_opset(m, target_ir_max=target_ir_max, target_opset=target_opset)
    ensure_dual_onnx_opset_ai_first(m, int(target_opset))
    set_meta_in_model(m, meta)
    print(f"[fix-fp32] After  IR={m.ir_version}, opset={_ops_to_str(m)}")
    save_onnx_external_onefile(m, output_file)
    onnx.checker.check_model(output_file)
    print(f"[fix-fp32] Saved: {output_file} (+{Path(output_file).with_suffix('.data').name})")


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
                model = onnx.load(model, load_external_data=True)
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


def quantize_int8_dynamic_no_data(input_fp32_onnx: str, output_file: str, target_ir_max: int = 9, target_opset: int = 17):
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
    ensure_dual_onnx_opset_ai_first(q, int(target_opset))

    save_onnx_single_file(q, output_file)
    onnx.checker.check_model(output_file)

    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    if Path(output_file).with_suffix(".data").exists():
        raise RuntimeError(f"int8 output still produced .data which is not allowed: {Path(output_file).with_suffix('.data')}")

    print(f"[int8] Saved: {output_file}")


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


def to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def unflatten_kv(flat: Tuple[torch.Tensor, ...], num_layers: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if len(flat) != 2 * num_layers:
        raise RuntimeError(f"expect {2*num_layers} past tensors, got {len(flat)}")
    out = []
    for i in range(num_layers):
        out.append((flat[2 * i], flat[2 * i + 1]))
    return tuple(out)


try:
    from transformers.cache_utils import Cache as HFCacheBase
except Exception:
    HFCacheBase = object


class OnnxConcatCache(HFCacheBase):
    def __init__(self, legacy_cache, seq_axis: int):
        self.seq_axis = int(seq_axis)
        self._k = [kv[0] for kv in legacy_cache]
        self._v = [kv[1] for kv in legacy_cache]
        if DEBUG_CACHE:
            print("[DEBUG] Cache init", self.seq_axis, tuple(self._k[0].shape), tuple(self._v[0].shape))

    @property
    def is_initialized(self):
        return True

    def __len__(self) -> int:
        return len(self._k)

    def __getitem__(self, layer_idx: int):
        return (self._k[layer_idx], self._v[layer_idx])

    def get_seq_length(self, layer_idx: int = 0):
        if torch.onnx.is_in_onnx_export():
            try:
                from torch.onnx import operators
                return operators.shape_as_tensor(self._k[layer_idx])[self.seq_axis]
            except Exception:
                return torch.as_tensor(int(self._k[layer_idx].shape[self.seq_axis]), device=self._k[layer_idx].device, dtype=torch.long)
        return int(self._k[layer_idx].shape[self.seq_axis])

    def get_max_length(self):
        return None

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        k = torch.cat([self._k[layer_idx], key_states], dim=self.seq_axis)
        v = torch.cat([self._v[layer_idx], value_states], dim=self.seq_axis)
        self._k[layer_idx] = k
        self._v[layer_idx] = v
        return k, v

    def to_legacy_cache(self):
        return tuple((self._k[i], self._v[i]) for i in range(len(self._k)))

    def reorder_cache(self, beam_idx: torch.Tensor):
        for i in range(len(self._k)):
            self._k[i] = self._k[i].index_select(0, beam_idx)
            self._v[i] = self._v[i].index_select(0, beam_idx)
        return self


class PrefillWrapper(torch.nn.Module):
    def __init__(self, model, num_layers: int, seq_axis: int):
        super().__init__()
        self.m = model
        self.num_layers = num_layers
        self.seq_axis = seq_axis

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        out = self.m(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True, return_dict=True)
        logits = out.logits
        legacy = to_legacy_cache(out.past_key_values)
        kv = []
        for (k, v) in legacy:
            kv.append(k)
            kv.append(v)
        return (logits, *kv)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, model, num_layers: int, seq_axis: int):
        super().__init__()
        self.m = model
        self.num_layers = num_layers
        self.seq_axis = seq_axis

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, *past_flat: torch.Tensor):
        legacy_in = unflatten_kv(past_flat, self.num_layers)
        cache = OnnxConcatCache(legacy_in, seq_axis=self.seq_axis)
        out = self.m(inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=cache, use_cache=True, return_dict=True)
        logits = out.logits
        legacy_out = cache.to_legacy_cache()
        kv_out = []
        for (k, v) in legacy_out:
            kv_out.append(k)
            kv_out.append(v)
        return (logits, *kv_out)


@torch.no_grad()
def infer_seq_axis_by_two_prefills(model, hidden_size: int, L1: int, L2: int) -> int:
    device = torch.device("cpu")
    model.eval().to(device)
    dt = next(model.parameters()).dtype

    x1 = torch.randn(1, L1, hidden_size, dtype=dt, device=device)
    am1 = torch.ones(1, L1, dtype=torch.int64, device=device)
    o1 = model(inputs_embeds=x1, attention_mask=am1, use_cache=True, return_dict=True)
    k1 = to_legacy_cache(o1.past_key_values)[0][0]
    s1 = tuple(int(i) for i in k1.shape)

    x2 = torch.randn(1, L2, hidden_size, dtype=dt, device=device)
    am2 = torch.ones(1, L2, dtype=torch.int64, device=device)
    o2 = model(inputs_embeds=x2, attention_mask=am2, use_cache=True, return_dict=True)
    k2 = to_legacy_cache(o2.past_key_values)[0][0]
    s2 = tuple(int(i) for i in k2.shape)

    diffs = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b]
    if len(diffs) == 1:
        return diffs[0]

    cand = []
    for i in diffs:
        if s1[i] == L1 and s2[i] == L2:
            cand.append(i)
    if len(cand) == 1:
        return cand[0]

    raise RuntimeError(f"Cannot infer seq axis. k0(L1)={s1}, k0(L2)={s2}, diffs={diffs}")


@torch.no_grad()
def probe_kv_layout(model, hidden_size: int, prefill_len: int) -> int:
    L1 = int(prefill_len)
    L2 = max(2, L1 - 3)
    seq_axis = infer_seq_axis_by_two_prefills(model, hidden_size, L1, L2)

    device = torch.device("cpu")
    model.eval().to(device)
    dt = next(model.parameters()).dtype

    x = torch.randn(1, L1, hidden_size, dtype=dt, device=device)
    am = torch.ones(1, L1, dtype=torch.int64, device=device)
    out = model(inputs_embeds=x, attention_mask=am, use_cache=True, return_dict=True)
    legacy = to_legacy_cache(out.past_key_values)

    x1 = torch.randn(1, 1, hidden_size, dtype=dt, device=device)
    am1 = torch.ones(1, L1 + 1, dtype=torch.int64, device=device)
    cache = OnnxConcatCache(legacy, seq_axis=seq_axis)
    _ = model(inputs_embeds=x1, attention_mask=am1, past_key_values=cache, use_cache=True, return_dict=True)

    print(f"KV probe OK. kv_seq_axis={seq_axis}")
    return seq_axis


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
        external_data=False,
    )


@torch.no_grad()
def export_prefill(model, hidden_size: int, num_layers: int, prefill_len: int, opset: int, out_path: str, kv_seq_axis: int):
    wrapped = PrefillWrapper(model, num_layers=num_layers, seq_axis=kv_seq_axis).eval()
    dt = next(model.parameters()).dtype
    x = torch.randn(1, prefill_len, hidden_size, dtype=dt)
    am = torch.ones(1, prefill_len, dtype=torch.int64)

    input_names = ["inputs_embeds", "attention_mask"]
    output_names = ["logits"]
    for i in range(num_layers):
        output_names += [f"present_key_{i}", f"present_value_{i}"]

    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(num_layers):
        dynamic_axes[f"present_key_{i}"] = {0: "batch", kv_seq_axis: "seq"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", kv_seq_axis: "seq"}

    print(f"Export prefill -> {out_path} (opset={opset}, kv_seq_axis={kv_seq_axis}, dtype={dt})")
    _torch_onnx_export(wrapped, (x, am), out_path, opset, input_names, output_names, dynamic_axes)


@torch.no_grad()
def export_decode(model, hidden_size: int, num_layers: int, past_len: int, opset: int, out_path: str, kv_seq_axis: int):
    wrapped = DecodeWrapper(model, num_layers=num_layers, seq_axis=kv_seq_axis).eval()
    dt = next(model.parameters()).dtype

    x = torch.randn(1, 1, hidden_size, dtype=dt)
    am = torch.ones(1, past_len + 1, dtype=torch.int64)

    xp = torch.randn(1, past_len, hidden_size, dtype=dt)
    amp = torch.ones(1, past_len, dtype=torch.int64)
    outp = model(inputs_embeds=xp, attention_mask=amp, use_cache=True, return_dict=True)
    legacy = to_legacy_cache(outp.past_key_values)

    past_flat = []
    for (k, v) in legacy:
        past_flat.append(k)
        past_flat.append(v)

    input_names = ["inputs_embeds", "attention_mask"]
    for i in range(num_layers):
        input_names += [f"past_key_{i}", f"past_value_{i}"]

    output_names = ["logits"]
    for i in range(num_layers):
        output_names += [f"present_key_{i}", f"present_value_{i}"]

    dynamic_axes = {
        "inputs_embeds": {0: "batch"},
        "attention_mask": {0: "batch", 1: "total_seq"},
        "logits": {0: "batch"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {0: "batch", kv_seq_axis: "past_seq"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch", kv_seq_axis: "past_seq"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch", kv_seq_axis: "total_seq"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", kv_seq_axis: "total_seq"}

    print(f"Export decode -> {out_path} (opset={opset}, kv_seq_axis={kv_seq_axis}, dtype={dt})")
    _torch_onnx_export(wrapped, (x, am, *past_flat), out_path, opset, input_names, output_names, dynamic_axes)


@torch.no_grad()
def verify_onnx(prefill_path: str, decode_path: str, hidden_size: int, vocab_size: int, num_layers: int):
    import onnxruntime as ort

    def _np_dtype(ort_type: str):
        if "float16" in ort_type:
            return np.float16
        if "float" in ort_type:
            return np.float32
        raise RuntimeError(f"Unsupported inputs_embeds type: {ort_type}")

    print("Verifying with onnxruntime...")

    sess_p = ort.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
    p_inputs = {i.name: i for i in sess_p.get_inputs()}

    T = 16
    x_dtype = _np_dtype(p_inputs["inputs_embeds"].type)
    x = np.random.randn(1, T, hidden_size).astype(x_dtype)
    am = np.ones((1, T), dtype=np.int64)
    out = sess_p.run(None, {"inputs_embeds": x, "attention_mask": am})
    logits = out[0]
    if not (logits.shape[0] == 1 and logits.shape[-1] == vocab_size):
        raise RuntimeError(f"Prefill logits shape mismatch: {logits.shape}")
    print("Prefill OK. logits:", logits.shape)

    sess_d = ort.InferenceSession(decode_path, providers=["CPUExecutionProvider"])
    d_inputs = {i.name: i for i in sess_d.get_inputs()}

    x1_dtype = _np_dtype(d_inputs["inputs_embeds"].type)
    x1 = np.random.randn(1, 1, hidden_size).astype(x1_dtype)
    am1 = np.ones((1, T + 1), dtype=np.int64)

    feed = {"inputs_embeds": x1, "attention_mask": am1}
    for i in range(num_layers):
        feed[f"past_key_{i}"] = out[1 + 2 * i]
        feed[f"past_value_{i}"] = out[1 + 2 * i + 1]

    out_d = sess_d.run(None, feed)
    logits_d = out_d[0]
    if not (logits_d.shape[0] == 1 and logits_d.shape[-1] == vocab_size):
        raise RuntimeError(f"Decode logits shape mismatch: {logits_d.shape}")
    print("Decode OK. logits:", logits_d.shape)


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


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pt", type=str, required=True)
    ap.add_argument("--llm-config-path", type=str, required=True)
    ap.add_argument("--output-filename", type=str, default="models/llm.onnx")
    ap.add_argument("--opset-version", type=int, default=18)
    ap.add_argument("--prefill-len", type=int, default=256)
    ap.add_argument("--past-len", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--quantize-int8", action="store_true")
    ap.add_argument("--target-opset", type=int, default=17)
    ap.add_argument("--target-ir-max", type=int, default=9)
    ap.add_argument("--keep-tmp", action="store_true")
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

    setattr(config, "_attn_implementation", "eager")
    print("Set config._attn_implementation = 'eager'")

    dtype = torch.float16
    print("Building HF model dtype=", dtype)
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

    kv_seq_axis = probe_kv_layout(model, hidden_size, prefill_len=min(8, args.prefill_len))

    root_dir = Path(args.output_filename).parent
    fp32_dir = root_dir / "llm_fp32"
    fp16_dir = root_dir / "llm_fp16"
    int8_dir = root_dir / "llm_int8"
    tmp_dir = root_dir / "_tmp_export_raw"

    fp32_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)
    if args.quantize_int8:
        int8_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_prefill_fp16 = str(fp16_dir / "llm_prefill.fp16.onnx")
    out_decode_fp16 = str(fp16_dir / "llm_decode.fp16.onnx")

    out_prefill_fp32 = str(fp32_dir / "llm_prefill.fp32.onnx")
    out_decode_fp32 = str(fp32_dir / "llm_decode.fp32.onnx")

    out_prefill_raw = str(tmp_dir / "llm_prefill.raw.onnx")
    out_decode_raw = str(tmp_dir / "llm_decode.raw.onnx")

    export_prefill(model, hidden_size, num_layers, int(args.prefill_len), int(args.opset_version), out_prefill_raw, kv_seq_axis)
    export_decode(model, hidden_size, num_layers, int(args.past_len), int(args.opset_version), out_decode_raw, kv_seq_axis)

    meta = {
        "model_type": "qwen3_causallm",
        "version": "1",
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "kv_seq_axis": kv_seq_axis,
    }

    fix_save_fp16_single(out_prefill_raw, out_prefill_fp16, meta, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
    fix_save_fp16_single(out_decode_raw, out_decode_fp16, meta, target_ir_max=args.target_ir_max, target_opset=args.target_opset)

    fix_save_fp32_external(out_prefill_raw, out_prefill_fp32, meta, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
    fix_save_fp32_external(out_decode_raw, out_decode_fp32, meta, target_ir_max=args.target_ir_max, target_opset=args.target_opset)

    out_prefill_int8 = None
    out_decode_int8 = None
    if args.quantize_int8:
        out_prefill_int8 = str(int8_dir / "llm_prefill.int8.onnx")
        out_decode_int8 = str(int8_dir / "llm_decode.int8.onnx")
        quantize_int8_dynamic_no_data(out_prefill_fp32, out_prefill_int8, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
        quantize_int8_dynamic_no_data(out_decode_fp32, out_decode_int8, target_ir_max=args.target_ir_max, target_opset=args.target_opset)

    if args.verify:
        verify_onnx(out_prefill_fp16, out_decode_fp16, hidden_size, vocab_size, num_layers)
        if args.quantize_int8:
            verify_onnx(out_prefill_int8, out_decode_int8, hidden_size, vocab_size, num_layers)

    if not args.keep_tmp:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    print("Done.")
    print("Generated:")
    print(" ", out_prefill_fp16)
    print(" ", out_decode_fp16)
    print(" ", out_prefill_fp32, "+", str(Path(out_prefill_fp32).with_suffix(".data")))
    print(" ", out_decode_fp32, "+", str(Path(out_decode_fp32).with_suffix(".data")))
    if args.quantize_int8:
        print(" ", out_prefill_int8)
        print(" ", out_decode_int8)


if __name__ == "__main__":
    torch.manual_seed(20251222)
    main()
