#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


class LLMOnnx:
    def __init__(self, filename: str, device: str = "cpu"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"ONNX model file not found: {filename}")

        filename_path = Path(filename)
        external_candidates = [
            filename_path.parent / f"{filename_path.name}.data",
            filename_path.parent / f"{filename_path.stem}.onnx_data",
        ]
        self.external_data_file = None
        for p in external_candidates:
            if p.exists():
                self.external_data_file = p
                break

        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        self.session_opts = session_opts

        try:
            if device == "cpu":
                use_providers = ["CPUExecutionProvider"]
            elif device == "cuda":
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    use_providers = ["CPUExecutionProvider"]
            else:
                use_providers = ["CPUExecutionProvider"]

            self.model = ort.InferenceSession(
                filename,
                sess_options=session_opts,
                providers=use_providers,
            )
        except Exception as e:
            raise

        meta = self.model.get_modelmeta().custom_metadata_map
        self.hidden_size = int(meta.get("hidden_size", 1024))
        self.vocab_size = int(meta.get("vocab_size", 151936))
        self.model_type = meta.get("model_type", "qwen3_causallm")

        model_inputs = self.model.get_inputs()
        model_outputs = self.model.get_outputs()
        self.num_inputs = len(model_inputs)
        self.num_outputs = len(model_outputs)

        if self.num_inputs == 0:
            raise RuntimeError("ONNX model has no inputs")

        self.input_name = model_inputs[0].name
        self.output_name = model_outputs[0].name

        # INT8 quantized models require float32 inputs
        is_int8_model = "int8" in filename.lower()
        
        first_input_type = str(model_inputs[0].type).lower()
        if is_int8_model:
            # INT8 models always use float32 inputs
            self.input_dtype = np.float32
        elif "float16" in first_input_type or "fp16" in first_input_type:
            self.input_dtype = np.float16
        elif "float32" in first_input_type or "float" in first_input_type:
            self.input_dtype = np.float32
        else:
            self.input_dtype = np.float32

    def __call__(self, inputs_embeds, attention_mask=None):
        import torch

        if isinstance(inputs_embeds, torch.Tensor):
            inputs_embeds = inputs_embeds.detach().cpu().numpy()
        elif not isinstance(inputs_embeds, np.ndarray):
            inputs_embeds = np.array(inputs_embeds)

        if np.any(np.isnan(inputs_embeds)) or np.any(np.isinf(inputs_embeds)):
            inputs_embeds = np.where(np.isfinite(inputs_embeds), inputs_embeds, 0.0)

        if self.input_dtype == np.float16:
            inputs_embeds = np.clip(inputs_embeds, -65504.0, 65504.0)
            inputs_embeds = inputs_embeds.astype(self.input_dtype, copy=False)
        else:
            inputs_embeds = inputs_embeds.astype(self.input_dtype, copy=False)

        if inputs_embeds.ndim != 3:
            raise ValueError(
                f"inputs_embeds must be 3-D (B,T,H), got shape {inputs_embeds.shape}"
            )

        ort_inputs = {self.input_name: inputs_embeds}

        if self.num_inputs > 1:
            if attention_mask is None:
                attention_mask = np.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]), dtype=np.int64
                )
            elif isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.detach().cpu().numpy()
            elif not isinstance(attention_mask, np.ndarray):
                attention_mask = np.array(attention_mask)
            attention_mask = attention_mask.astype(np.int64, copy=False)
            second_name = self.model.get_inputs()[1].name
            ort_inputs[second_name] = attention_mask

        logits = self.model.run([self.output_name], ort_inputs)[0]
        
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            logits = np.where(np.isfinite(logits), logits, -1e10)
        
        return logits
