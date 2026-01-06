# FunASR-Nano ONNX

## Environment
Python ≥ 3.8; install deps from `requirements.txt`.

## Get models
- Download ready-made ONNX models:  
  `modelscope download --model zengshuishui/FunASR-nano-onnx --local_dir models`

## Export (only if you need to re-export from model.pt)
1. Edit `scripts/run.sh` and set your paths:  
   - `--model-pt .../model.pt`  
   - `--llm-config-path .../Qwen3-0.6B`  
   - `--output-root ../models` (change if you want another output dir)
2. Run:
   ```bash
   cd scripts
   bash run.sh
   ```
   ONNX files will be placed in `../models/`.

## Decode demo
With downloaded or exported models, run:
```bash
bash decode.sh
```
`decode.sh` is a minimal demo, using INT8/FP32 models for quick sanity check.

## C++ inference
See detailed C++ examples in sherpa-onnx: <https://github.com/k2-fsa/sherpa-onnx>

## Notes
- `run.sh`: one-click export (edit paths first).  
- `decode.sh`: demo decode to verify models.  
