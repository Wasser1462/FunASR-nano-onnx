#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class EmbeddingOnnx:
    def __init__(self, filename: str, device: str = "cpu"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"ONNX embedding model file not found: {filename}")

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
        self.vocab_size = int(meta.get("vocab_size", 151936))
        self.hidden_size = int(meta.get("hidden_size", 1024))
        self.model_type = meta.get("model_type", "embedding_layer")

        model_inputs = self.model.get_inputs()
        model_outputs = self.model.get_outputs()

        if len(model_inputs) == 0:
            raise RuntimeError("ONNX embedding model has no inputs")
        if len(model_outputs) == 0:
            raise RuntimeError("ONNX embedding model has no outputs")

        self.input_name = model_inputs[0].name
        self.output_name = model_outputs[0].name

        first_output_type = str(model_outputs[0].type).lower()
        self.input_dtype = np.int64

        is_int8_model = "int8" in filename.lower()
        if is_int8_model:
            self.output_dtype = np.float32
        elif "float16" in first_output_type or "fp16" in first_output_type:
            self.output_dtype = np.float16
        elif "float32" in first_output_type or "float" in first_output_type:
            self.output_dtype = np.float32
        else:
            self.output_dtype = np.float32
    
    def __call__(self, input_ids):
        import torch

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().numpy()
        elif isinstance(input_ids, list):
            input_ids = np.array(input_ids, dtype=np.int64)
            if input_ids.ndim == 1:
                input_ids = input_ids[None, :]
        elif not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)

        input_ids = input_ids.astype(np.int64, copy=False)

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2-D (batch_size, seq_length), got shape {input_ids.shape}"
            )

        input_ids = np.clip(input_ids, 0, self.vocab_size - 1)

        ort_inputs = {self.input_name: input_ids}
        embeddings = self.model.run([self.output_name], ort_inputs)[0]

        if embeddings.dtype != self.output_dtype:
            embeddings = embeddings.astype(self.output_dtype, copy=False)

        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            embeddings = np.where(np.isfinite(embeddings), embeddings, 0.0)

        return embeddings

