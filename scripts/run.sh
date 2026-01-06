mkdir -p ../models

model_pt_path=/path/to/Fun-ASR-Nano-2512/model.pt
llm_config_path=/path/to/Qwen3-0.6B

python export_encoder_adaptor_onnx.py \
    --model-pt $model_pt_path \
    --output-filename ../models/encoder_adaptor.onnx \
    --opset-version 18

python export_embedding_onnx.py \
    --llm-config-path $llm_config_path \
    --output-filename ../models/embedding.onnx \
    --opset-version 18 \
    --verify

# export DEBUG_EXPORT=1
# export DEBUG_QWEN=1

python export_llm_onnx.py \
  --model-pt $model_pt_path \
  --llm-config-path $llm_config_path \
  --output-root ../models \
  --opset-version 17 \
  --seq-len 256 \
  --past-len 256 \
  --verify

bash fix_all_models.sh

rm -rf ../models/*.backup

echo "All models have been exported !"