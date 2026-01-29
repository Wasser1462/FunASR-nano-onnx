# FunASR-Nano ONNX

## Environment
Python ≥ 3.8; install deps from `requirements.txt`.

## Get models
- Download ready-made ONNX models:  
  `modelscope download --model zengshuishui/FunASR-nano-onnx --local_dir models`

## Export (only if you need to re-export from model.pt)
1. Edit `scripts/run.sh` and set your paths:  
   - `fun_asr_path=/path/to/Fun-ASR-Nano-2512`  

2. Run:
   ```bash
   cd scripts
   ```
   ```bash
   bash run.sh
   ```
   ONNX files will be placed in `../models/`.

## Decode demo
With downloaded or exported models, run:
```bash
bash decode.sh
```
`decode.sh` is a minimal demo, using INT8/FP32 models for quick sanity check.

## Streaming ASR (WebSocket Server)
Run the real-time streaming ASR server:
```bash
cd streaming_fun_asr
```
```bash
python realtime_ws_server.py
```
Supports INT8 models (CPU) and FP32 models (CUDA), auto-detected. Open `http://localhost:8000` for the demo page.

## C++ inference
See detailed C++ examples in sherpa-onnx: <https://github.com/k2-fsa/sherpa-onnx>

## Notes
- `run.sh`: one-click export (edit paths first).  
- `decode.sh`: demo decode to verify models.
- **Project Refactoring**: The project has been refactored. The original main branch has been backed up to `backup/main-2026-01-06`.  
