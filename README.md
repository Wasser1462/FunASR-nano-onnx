# FunASR-Nano ONNX

ONNX export and inference implementation for FunASR-Nano model.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- ONNX Runtime >= 1.15
- transformers
- funasr (for feature extraction)
- modelscope (for downloading models)

Install dependencies:

```bash
pip install -r requirements.txt
pip install modelscope
```

## Quick Start

### 1. Download Models

Download pre-trained ONNX models from ModelScope to the `models/` directory:

```bash
modelscope download --model zengshuishui/FunASR-nano-onnx --output_dir models
```

After downloading, the `models/` directory will contain:
- `encoder_adaptor.onnx` and `encoder_adaptor.onnx.data`
- `llm.onnx` and `llm.onnx.data`
- `encoder_adaptor.int8.onnx` and `llm.int8.onnx` (INT8 quantized versions)
- `embedding.onnx` and `embedding.int8.onnx`

### 2. Run Inference

**With ONNX Embedding Model (Recommended)**:

```bash
python inference.py \
    --encoder-adaptor-model models/encoder_adaptor.onnx \
    --llm-model models/llm.onnx \
    --llm-tokenizer models/Qwen3-0.6B \
    --embedding-model models/embedding.onnx \
    --wave examples/zh.mp3 \
    --prompt "语音转写：" \
    --max-new-tokens 512 \
    --device auto
```

**Without ONNX Embedding Model (requires model.safetensors)**:

```bash
python inference.py \
    --encoder-adaptor-model models/encoder_adaptor.onnx \
    --llm-model models/llm.onnx \
    --llm-tokenizer models/Qwen3-0.6B \
    --wave examples/zh.mp3 \
    --prompt "语音转写：" \
    --max-new-tokens 512 \
    --device auto
```

**Parameters**:
- `--device`: Inference device, options: `cpu`, `cuda`, or `auto` (default: `auto`, LLM uses CPU by default due to CUDA float16 issues)
- `--embedding-model`: Path to ONNX embedding model (optional, if not provided, will use PyTorch model from `--llm-tokenizer`)
- `--seed`: Random seed for reproducible results (default: 42)
- `--temperature`: Sampling temperature (default: 0.3)
- `--top-p`: Top-p (nucleus) sampling threshold (default: 0.8)

### 3. Export ONNX Models (Optional)

If you need to export ONNX models from the original model.pt:

#### Export Encoder+Adaptor

```bash
python scripts/export_encoder_adaptor_onnx.py \
    --model-pt /path/to/model.pt \
    --output-filename models/encoder_adaptor.onnx \
    --opset-version 18
```

#### Export LLM

```bash
python scripts/export_llm_onnx.py \
    --model-pt /path/to/model.pt \
    --llm-config-path /path/to/Qwen3-0.6B \
    --output-filename models/llm.onnx \
    --opset-version 18
```

#### Export Embedding Layer (Optional, Recommended)

To avoid loading the full PyTorch model during inference, you can export the embedding layer to ONNX:

```bash
python scripts/export_embedding_onnx.py \
    --llm-config-path /path/to/Qwen3-0.6B \
    --output-filename models/embedding.onnx \
    --opset-version 18 \
    --verify
```

The `--verify` flag will automatically verify the exported model by comparing with the PyTorch model. This script will:
1. Check if `model.safetensors` exists in the LLM config directory
2. Export the embedding layer to ONNX format
3. Create INT8 quantized version 
4. Verify the exported model (if `--verify` is used)

**Note**: The embedding ONNX model eliminates the need for `model.safetensors` during inference, reducing memory usage and startup time. The INT8 quantized version further reduces model size while maintaining accuracy.

## Model Description

### Encoder+Adaptor Model

- **Input**: Audio features `(batch, time, 560)`
- **Output**: LLM embeddings `(batch, time, 1024)`
- **Supports dynamic sequence length**

### LLM Model

- **Input**: 
  - `inputs_embeds`: `(batch, sequence_length, 1024)`
  - `attention_mask`: `(batch, sequence_length)`
- **Output**: `logits`: `(batch, sequence_length, vocab_size)`
- **Supports dynamic sequence length**

### Embedding Model (Optional)

- **Input**: `input_ids`: `(batch, sequence_length)` - Token IDs (int64)
- **Output**: `embeddings`: `(batch, sequence_length, 1024)` - Token embeddings
- **Supports dynamic sequence length**
- **Purpose**: Converts token IDs to embeddings, eliminating the need for full PyTorch model during inference

### GPU Acceleration

Make sure `onnxruntime-gpu` is installed:

```bash
pip install onnxruntime-gpu
```

Note: Due to CUDA provider issues with float16, the LLM model uses CPU by default. The Encoder+Adaptor model can use GPU if available.

Use GPU for Encoder+Adaptor (LLM uses CPU):

```bash
python inference.py \
    --encoder-adaptor-model models/encoder_adaptor.onnx \
    --llm-model models/llm.onnx \
    --llm-tokenizer models/Qwen3-0.6B \
    --wave examples/zh.mp3 \
    --device cuda
```

Use CPU for all models:

```bash
python inference.py \
    --encoder-adaptor-model models/encoder_adaptor.onnx \
    --llm-model models/llm.onnx \
    --llm-tokenizer models/Qwen3-0.6B \
    --wave examples/zh.mp3 \
    --device cpu
```

## License

Please refer to the license of the original FunASR project.

## Acknowledgments

- Based on the [FunASR](https://github.com/alibaba-damo-academy/FunASR) project.
- Code structure and ONNX export implementation inspired by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
