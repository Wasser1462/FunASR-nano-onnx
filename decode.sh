set -e

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

echo "Decoding with FP32 models on GPU (batch)"
python3 batch_inference.py \
  --encoder-adaptor-model models/encoder_adaptor.onnx \
  --embedding-model models/embedding.onnx \
  --llm-model models/llm_fp32/llm.fp32.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/zh.wav examples/en.wav examples/ja.wav examples/ko.wav examples/yue.wav\
  --prompt "语音转写：" \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --top-p 1.0 \
  --llm-device cuda \
  --embedding-device cuda \
  --encoder-device cuda

echo "Decoding with INT8 models on CPU (batch)"
python3 batch_inference.py \
  --encoder-adaptor-model models/encoder_adaptor.onnx \
  --embedding-model models/embedding.onnx \
  --llm-model models/llm_int8/llm.int8.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/zh.wav examples/en.wav examples/ja.wav examples/ko.wav examples/yue.wav \
  --prompt "语音转写：" \
  --max-new-tokens 128 \
  --llm-device cpu \
  --embedding-device cpu \
  --encoder-device cpu

echo "Decoding with FP32 models on GPU (hotword)"
python hotword_inference.py \
  --encoder-adaptor-model models/encoder_adaptor.onnx \
  --embedding-model models/embedding.onnx \
  --llm-model models/llm_fp32/llm.fp32.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/zh.wav \
  --hotword-file hotwords/hotwords.txt \
  --rectify-file hotwords/hot-rectify.txt

echo "Decoding with INT8 models on CPU (hotword version 2)"
python3 hotword_inference_2.py \
  --encoder-adaptor-model models/encoder_adaptor.int8.onnx \
  --embedding-model models/embedding.int8.onnx \
  --llm-model models/llm_int8/llm.int8.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/zh.wav \
  --language 中文 \
  --hotwords "开放时间,早上酒店" \
  --no-itn


echo "Decoding with FP32 models on GPU (VAD)"
python3 vad_inference.py \
  --encoder-adaptor-model models/encoder_adaptor.onnx \
  --embedding-model models/embedding.onnx \
  --llm-model models/llm_fp32/llm.fp32.onnx \
  --llm-tokenizer models/Qwen3-0.6B \
  --wave examples/song.wav \
  --vad-model models/silero_vad.onnx \
  --vad-pad-sec 0.30 \
  --vad-merge-gap-sec 0.20 \
  --vad-min-seg-sec 0.20 \
  --vad-max-seg-sec 20.0 \
  --vad-split-overlap-sec 0.20


echo "All Done!"