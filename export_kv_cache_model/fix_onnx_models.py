#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, Optional, List

import numpy as np
import onnx
from onnx import helper, numpy_helper


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
    model.opset_import.extend(
        [
            _make_opsetid("ai.onnx", int(target_opset)),
            _make_opsetid("", int(target_opset)),
        ]
    )
    model.opset_import.extend(others)


def force_ir_opset(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    if int(model.ir_version) > int(target_ir_max):
        model.ir_version = int(target_ir_max)
    ensure_dual_onnx_opset_ai_first(model, int(target_opset))


def _all_value_uses(model: onnx.ModelProto) -> Dict[str, int]:
    uses: Dict[str, int] = {}
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
            modified = True
    return modified


def _find_valueinfo_rank(model: onnx.ModelProto, name: str) -> Optional[int]:
    def _rank_from_vi(vi) -> Optional[int]:
        t = vi.type.tensor_type
        if not t or not t.HasField("shape"):
            return None
        dims = t.shape.dim
        if dims is None:
            return None
        return len(dims)

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.name == name:
            r = _rank_from_vi(vi)
            if r is not None and r > 0:
                return r

    arr = _get_const(model, name)
    if arr is not None:
        return int(np.array(arr).ndim)
    return None


def fix_pad_axes_input_op18_to_op17(model: onnx.ModelProto) -> bool:
    modified = False
    maybe_delete: List[str] = []
    for node in model.graph.node:
        if node.op_type != "Pad":
            continue
        if len(node.input) != 4:
            continue
        data_name, pads_name, const_name, axes_name = node.input
        if not axes_name:
            continue
        pads_arr = _get_const(model, pads_name)
        axes_arr = _get_const(model, axes_name)
        if pads_arr is None or axes_arr is None:
            continue

        pads = np.array(pads_arr).astype(np.int64).reshape(-1)
        axes = np.array(axes_arr).astype(np.int64).reshape(-1)
        k = int(axes.shape[0])
        if k <= 0:
            continue
        if pads.shape[0] != 2 * k:
            continue

        rank = _find_valueinfo_rank(model, data_name)
        if rank is None:
            rank = int(np.max(axes)) + 1
        rank = max(rank, int(np.max(axes)) + 1)

        pads_full = np.zeros((2 * rank,), dtype=np.int64)
        begin = pads[:k]
        end = pads[k:]
        for i, ax in enumerate(axes.tolist()):
            if ax < 0:
                ax = rank + ax
            if ax < 0 or ax >= rank:
                continue
            pads_full[ax] = begin[i]
            pads_full[rank + ax] = end[i]

        base = f"{(node.name or 'Pad')}_pads_full"
        new_name = base
        existing = {init.name for init in model.graph.initializer}
        t = 0
        while new_name in existing:
            t += 1
            new_name = f"{base}_{t}"

        model.graph.initializer.append(numpy_helper.from_array(pads_full, name=new_name))
        node.input[1] = new_name
        node.input.pop()

        maybe_delete.extend([pads_name, axes_name])
        modified = True

    if modified and maybe_delete:
        uses2 = _all_value_uses(model)
        for n in maybe_delete:
            if uses2.get(n, 0) <= 1:
                _remove_initializer(model, n)
                _remove_constant_node_output(model, n)
    return modified


def fix_model_inplace(model: onnx.ModelProto, target_ir_max: int = 9, target_opset: int = 17):
    fix_split_num_outputs(model)
    fix_reduce_axes_input_to_attr(model)
    fix_pad_axes_input_op18_to_op17(model)
    force_ir_opset(model, target_ir_max=target_ir_max, target_opset=target_opset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--target-opset", type=int, default=17)
    ap.add_argument("--target-ir-max", type=int, default=9)
    args = ap.parse_args()

    m = onnx.load(args.input, load_external_data=False)
    print(f"Before IR={m.ir_version}, opset={_ops_to_str(m)}")
    fix_model_inplace(m, target_ir_max=args.target_ir_max, target_opset=args.target_opset)
    print(f"After  IR={m.ir_version}, opset={_ops_to_str(m)}")

    try:
        onnx.checker.check_model(m)
        print("checker: OK")
    except Exception as e:
        print("checker: FAIL:", str(e))

    onnx.save(m, args.output)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
