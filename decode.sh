echo "Decoding with INT8 models on CPU"

python3 inference.py \
  --encoder-adaptor-model models/encoder_adaptor.int8.onnx \
  --embedding-model models/embedding.int8.onnx \
  --llm-model models/llm_int8/llm.int8.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/zh.wav \
  --prompt "语音转写：" \
  --encoder-device cpu \
  --llm-device cpu \
  --embedding-device cpu \
  --temperature 0.0

echo "Decoding with FP32 models on GPU"

python3 inference.py \
  --encoder-adaptor-model models/encoder_adaptor.onnx \
  --embedding-model models/embedding.onnx \
  --llm-model models/llm_fp32/llm.fp32.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/zh.wav \
  --prompt "语音转写：" \
  --encoder-device cuda \
  --llm-device cuda \
  --embedding-device cuda \
  --temperature 0.0

echo "Done!"