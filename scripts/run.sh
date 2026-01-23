mkdir -p ../models

model_pt_path=/path/to/Fun-ASR-Nano-2512/model.pt
llm_config_path=/path/to/Fun-ASR-Nano-2512/Qwen3-0.6B

python export_encoder_adaptor_onnx.py \
    --model-pt $model_pt_path \
    --output-filename ../models/encoder_adaptor.onnx \

python export_embedding_onnx.py \
    --llm-config-path $llm_config_path \
    --model-pt $model_pt_path \
    --output-filename ../models/embedding.onnx \
    --opset-version 17 \
    --verify

# export DEBUG_EXPORT=1
# export DEBUG_QWEN=1

# rm -rf ../models/llm_*

python export_llm_onnx.py \
  --model-pt $model_pt_path \
  --llm-config-path $llm_config_path \
  --output-root ../models \
  --opset-version 17 \
  --seq-len 256 \
  --past-len 256 \
  --verify

# NOTE:
# Some AVX2-only CPU environments may produce incorrect / unstable outputs with the original INT8 LLM
# (e.g., repeated tokens or garbled text) due to specific ORT INT8 kernel paths.
# This exporter generates AVX2-safe INT8 compatibility variants (llm_int8_compat) that have been
# validated in Colab, so users on AVX2-only machines can replace llm_int8/llm.int8.onnx with a compat one.
#
# python export_llm_onnx_u8u8.py \
#   --model-pt $model_pt_path \
#   --llm-config-path $llm_config_path \
#   --output-root ../models \
#   --opset-version 17 \
#   --seq-len 256 \
#   --past-len 256 \
#   --verify


bash fix_all_models.sh

rm -rf ../models/*.backup

echo "All models have been exported !"