# FunASR-nano KV Cache Model Export

This directory contains scripts to export FunASR-nano models to ONNX format with KV cache support for efficient autoregressive inference.

## Overview

The export process generates three types of ONNX models:

- **encoder_adaptor.onnx**: Converts audio features to embeddings compatible with the LLM
- **embedding.onnx**: Converts token IDs to embeddings for text prompts
- **llm_prefill.onnx** + **llm_decode.onnx**: Split LLM model for efficient KV cache-based autoregressive decoding

## Quick Start

Export all models:

```bash
bash run.sh
```

## Why Split LLM into Prefill and Decode Models?

We split the LLM into two ONNX graphs (`llm_prefill.onnx` and `llm_decode.onnx`) to match the two distinct phases of autoregressive inference:

### Prefill Phase (`llm_prefill.onnx`)

Runs **once** on a sequence of embeddings (prompt/audio+prompt tokens) to produce:

- Logits for the whole sequence
- Initial KV cache (past_key/past_value) for every transformer layer

### Decode Phase (`llm_decode.onnx`)

Runs **repeatedly** (token-by-token) with:

- The embedding of the next token (usually length = 1)
- The existing KV cache

And returns:

- Logits for the next token
- Updated KV cache (present_key/present_value)

### Benefits of This Split

This architecture makes the runtime interface stable and efficient:

- **Prefill handles "many tokens at once"** - Processes the entire context in a single forward pass
- **Decode handles "one token per step"** - Efficient incremental generation with cache reuse

## Export Process Details

### 1. Load HuggingFace LLM Weights

- Extract only `llm.*` tensors from `model.pt` and load them into the HuggingFace model

### 2. Make the Model ONNX-Friendly

- Force eager attention (or a compatible attention path)
- Disable sliding-window / incompatible cache behaviors
- Patch causal mask generation to avoid exporting problematic dynamic control logic

### 3. Probe KV Layout

Automatically infer which tensor axis is the "sequence length axis" inside KV cache (`kv_seq_axis`), because different implementations store KV in different layouts.

### 4. Export Two Graphs

- **Prefill graph**: Outputs logits + all layers' `present_key_i`/`present_value_i`
- **Decode graph**: Takes all layers' `past_key_i`/`past_value_i` and outputs updated `present_*`

### 5. Fix ONNX Graph Compatibility

- Convert `Reduce*` ops that incorrectly have axes as a second input into the standard attribute form (required by many runtimes/opsets)
- Remove problematic attributes like `Split.num_outputs` when it breaks checker/ORT compatibility
- Force a compatible opset/IR target (e.g., opset 17 and IR <= 9)
- Ensure opset imports include both `"ai.onnx"` and `""` domains in a safe order

### 6. Produce Final Models

- **Full precision models**: `llm_prefill.onnx`, `llm_decode.onnx` (FP32)
- **Quantized models**: `llm_prefill.int8.onnx`, `llm_decode.int8.onnx` (INT8 dynamic quantization)
