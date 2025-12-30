mkdir -p models

python export_encoder_adaptor_onnx.py \
    --model-pt /home/ec_user/workspace/work/test/Fun-ASR-Nano-2512/model.pt \
    --output-filename models/encoder_adaptor.onnx \
    --opset-version 18

python export_embedding_onnx.py \
    --llm-config-path /home/ec_user/workspace/work/test/Fun-ASR-Nano-2512/Qwen3-0.6B \
    --output-filename models/embedding.onnx \
    --opset-version 18 \
    --verify

python export_llm_onnx.py \
  --model-pt /home/ec_user/workspace/work/test/Fun-ASR-Nano-2512/model.pt \
  --llm-config-path /home/ec_user/workspace/work/test/Fun-ASR-Nano-2512/Qwen3-0.6B \
  --output-filename models/llm.onnx \
  --opset-version 18 \
  --prefill-len 256 --past-len 256 --verify \
  --quantize-int8

bash fix_all_models.sh

rm -rf models/*.backup

echo "Done!"