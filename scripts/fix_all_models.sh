#!/bin/bash

set -e

MODELS_DIR="${1:-../models}"

echo "Fixing ONNX models in ${MODELS_DIR}..."

MODELS=(
    "encoder_adaptor.onnx"
    "encoder_adaptor.int8.onnx"
    "embedding.onnx"
    "embedding.int8.onnx"
)

for model in "${MODELS[@]}"; do
    input_file="${MODELS_DIR}/${model}"
    if [ -f "$input_file" ]; then
        echo "Processing: $input_file"
        cp "$input_file" "${input_file}.backup"
        python3 fix_onnx_models.py \
            --input "$input_file" \
            --output "$input_file"
        echo "Done!"
    else
        echo "Warning: $input_file not found, skipping..."
    fi
done
