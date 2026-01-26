#!/usr/bin/env python3
#
# Copyright (c)  2025  zengyw
import asyncio
import json
import os
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

import onnxruntime as ort

import sys
sys.path.insert(0, os.path.dirname(__file__))
from streaming_fun_asr_llm import FunASRCore, FunASRSession, StreamingConfig

from typing import Dict, List, Tuple
import re
from dataclasses import dataclass
from transformers import AutoTokenizer


def pick_providers(device: str):
    providers = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]


@dataclass
class VADConfig:
    model_path: str
    device: str = "cpu"
    sample_rate: int = 16000
    threshold: float = 0.3
    window_size: int = 512
    hop_size: int = 160
    prepad_ms: int = 200
    postpad_ms: int = 200
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.5


class SileroVADCore:
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(cfg.model_path, sess_options=so, providers=pick_providers(cfg.device))
        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()
        self.in_names = [i.name for i in ins]
        self.out_names = [o.name for o in outs]
        self.in_x = self.in_names[0]
        self.in_sr = None
        for n in self.in_names:
            if "sr" in n.lower():
                self.in_sr = n
                break
        self.in_h = None
        self.in_c = None
        for n in self.in_names:
            if n.lower() == "h":
                self.in_h = n
            if n.lower() == "c":
                self.in_c = n
        self.out_p = self.out_names[0]
        self.out_hn = None
        self.out_cn = None
        for n in self.out_names:
            if n.lower() in ("hn", "h_n", "h_out", "h"):
                self.out_hn = n
            if n.lower() in ("cn", "c_n", "c_out", "c"):
                self.out_cn = n

    def init_state(self):
        if self.in_h is None or self.in_c is None:
            return None, None
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        return h, c

    def infer(self, frame: np.ndarray, h, c):
        x = frame.astype(np.float32).reshape(1, -1)
        feed = {self.in_x: x}
        if self.in_sr is not None:
            feed[self.in_sr] = np.array([self.cfg.sample_rate], dtype=np.int64)
        if self.in_h is not None and self.in_c is not None and h is not None and c is not None:
            feed[self.in_h] = h
            feed[self.in_c] = c
        outs = self.sess.run(None, feed)
        p = float(np.array(outs[0]).reshape(-1)[0])
        hn = None
        cn = None
        if self.out_hn is not None and self.out_cn is not None:
            idx_hn = self.out_names.index(self.out_hn)
            idx_cn = self.out_names.index(self.out_cn)
            hn = outs[idx_hn]
            cn = outs[idx_cn]
        elif len(outs) >= 3 and self.in_h is not None and self.in_c is not None:
            hn = outs[1]
            cn = outs[2]
        return p, hn, cn


class VADStream:
    def __init__(self, core: SileroVADCore, cfg: VADConfig):
        self.core = core
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.sample_index = 0
        self.in_speech = False
        self._buf = np.zeros((0,), dtype=np.float32)
        self._prepad = np.zeros((0,), dtype=np.float32)
        self._speech_run = 0
        self._silence_run = 0
        self._h, self._c = self.core.init_state()
        self._speech_start_sample = 0

    def feed(self, chunk_f32: np.ndarray) -> List[Tuple[Dict, np.ndarray, bool, int, int]]:
        cfg = self.cfg
        chunk_f32 = np.asarray(chunk_f32, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, chunk_f32], axis=0)

        prepad_keep = int(cfg.sample_rate * cfg.prepad_ms / 1000)
        min_speech_frames = max(1, int(cfg.min_speech_duration * cfg.sample_rate / cfg.hop_size))
        min_silence_frames = max(1, int(cfg.min_silence_duration * cfg.sample_rate / cfg.hop_size))

        events: List[Tuple[Dict, np.ndarray, bool, int, int]] = []
        produced = np.zeros((0,), dtype=np.float32)

        while self._buf.shape[0] >= cfg.window_size:
            frame = self._buf[:cfg.window_size]
            p, self._h, self._c = self.core.infer(frame, self._h, self._c)
            is_speech = (p >= cfg.threshold)

            if not self.in_speech:
                self._prepad = np.concatenate([self._prepad, self._buf[:cfg.hop_size]], axis=0)
                if self._prepad.shape[0] > prepad_keep:
                    self._prepad = self._prepad[-prepad_keep:]
            else:
                produced = np.concatenate([produced, self._buf[:cfg.hop_size]], axis=0)

            if is_speech:
                self._speech_run += 1
                self._silence_run = 0
            else:
                self._silence_run += 1
                if not self.in_speech:
                    self._speech_run = 0

            if (not self.in_speech) and is_speech and self._speech_run >= min_speech_frames:
                self.in_speech = True
                self._speech_start_sample = self.sample_index
                start_audio = np.concatenate([self._prepad, self._buf[:cfg.hop_size]], axis=0)
                self._prepad = np.zeros((0,), dtype=np.float32)
                s0 = max(0, self._speech_start_sample - len(start_audio))
                s1 = self.sample_index + cfg.hop_size
                events.append(({"start": s0}, start_audio, False, s0, s1))

            if self.in_speech and (not is_speech) and self._silence_run >= min_silence_frames:
                self.in_speech = False
                end_audio = np.concatenate([produced, self._buf[:cfg.hop_size]], axis=0)
                produced = np.zeros((0,), dtype=np.float32)
                s0 = self._speech_start_sample
                s1 = self.sample_index + cfg.hop_size
                events.append(({"end": s1}, end_audio, True, s0, s1))

            self._buf = self._buf[cfg.hop_size:]
            self.sample_index += cfg.hop_size

        if self.in_speech and produced.shape[0] > 0:
            s0 = self._speech_start_sample
            s1 = self.sample_index
            events.append(({}, produced, False, s0, s1))

        return events

    def force_end(self):
        if not self.in_speech:
            return None
        self.in_speech = False
        s1 = self.sample_index
        return {"end": s1}, np.zeros((0,), dtype=np.float32), True, self._speech_start_sample, s1


class TranscriptionChunk(BaseModel):
    timestamps: list = []
    raw_text: str
    final_text: Optional[str] = None
    delta: Optional[str] = None


class TranscriptionResponse(BaseModel):
    type: str = "TranscriptionResponse"
    id: int
    begin_at: float
    end_at: Optional[float] = None
    data: TranscriptionChunk
    is_final: bool
    session_id: Optional[str] = None


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool


_default_models_dir = os.path.join(os.path.dirname(__file__), "..", "models")


class Config(BaseSettings, cli_parse_args=True, cli_use_class_docs_for_groups=True):
    HOST: str = Field("127.0.0.1")
    PORT: int = Field(8000)
    DEBUG: bool = Field(False)

    ENCODER_ADAPTOR_MODEL: str = Field(os.path.join(_default_models_dir, "encoder_adaptor.int8.onnx"))
    EMBEDDING_MODEL: str = Field(os.path.join(_default_models_dir, "embedding.int8.onnx"))
    LLM_MODEL: str = Field(os.path.join(_default_models_dir, "llm_int8", "llm.int8.onnx"))
    LLM_TOKENIZER: str = Field(os.path.join(_default_models_dir, "Qwen3-0.6B"))

    VAD_MODEL: str = Field(os.path.join(_default_models_dir, "silero_vad.onnx"))
    DEVICE: str = Field("cuda")
    SAMPLERATE: int = Field(16000)

    CHUNK_DURATION: float = Field(0.1)
    VAD_THRESHOLD: float = Field(0.3)
    VAD_MIN_SILENCE_DURATION: float = Field(0.4)
    VAD_MIN_SPEECH_DURATION: float = Field(0.25)
    VAD_PREPAD_MS: int = Field(200)
    VAD_POSTPAD_MS: int = Field(200)
    VAD_WINDOW_SIZE: int = Field(512)
    VAD_HOP_SIZE: int = Field(160)

    MAX_NEW_TOKENS_PER_CHUNK: int = Field(20)
    AUDIO_WINDOW_MS: int = Field(6000)
    WARMUP_MS: int = Field(800)
    MIN_COMMIT_AUDIO_MS: int = Field(2500)
    FINAL_DECODE_ON_END: bool = Field(True)
    FINAL_MAX_NEW_TOKENS: int = Field(256)

    SYSTEM_PROMPT: str = Field("You are a helpful assistant.")
    PROMPT_ZH_STREAMING: str = Field("流式语音转写：")
    PROMPT_ZH_OFFLINE: str = Field("语音转写：")


config = Config()

logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG" if config.DEBUG else "INFO",
    colorize=True,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GLOBAL_CORE: Optional[FunASRCore] = None
GLOBAL_VAD_CORE: Optional[SileroVADCore] = None
CORE_LOCK: Optional[asyncio.Lock] = None


@app.on_event("startup")
async def startup_event():
    global GLOBAL_CORE, GLOBAL_VAD_CORE, CORE_LOCK
    # Determine runtime based on device and model path
    # If device is cpu or model is int8, use cpu runtime
    runtime = "cpu" if config.DEVICE == "cpu" or "int8" in config.LLM_MODEL.lower() else "auto"
    
    core_cfg = StreamingConfig(
        encoder_adaptor_model=config.ENCODER_ADAPTOR_MODEL,
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.LLM_MODEL,
        llm_tokenizer=config.LLM_TOKENIZER,
        encoder_device=config.DEVICE,
        embedding_device=config.DEVICE,
        llm_device=config.DEVICE,
        runtime=runtime,  # Explicitly set runtime to avoid auto-detection issues
        sample_rate=config.SAMPLERATE,
        prompt_zh_streaming=config.PROMPT_ZH_STREAMING,
        prompt_zh_offline=config.PROMPT_ZH_OFFLINE,
        system_prompt=config.SYSTEM_PROMPT,
        max_new_tokens_per_chunk=config.MAX_NEW_TOKENS_PER_CHUNK,
        audio_window_ms=config.AUDIO_WINDOW_MS,
        warmup_ms=config.WARMUP_MS,
        min_commit_audio_ms=config.MIN_COMMIT_AUDIO_MS,
        final_decode_on_end=config.FINAL_DECODE_ON_END,
        final_max_new_tokens=config.FINAL_MAX_NEW_TOKENS,
    )
    GLOBAL_CORE = FunASRCore(core_cfg)

    vad_cfg = VADConfig(
        model_path=config.VAD_MODEL,
        device=config.DEVICE,
        sample_rate=config.SAMPLERATE,
        threshold=config.VAD_THRESHOLD,
        window_size=config.VAD_WINDOW_SIZE,
        hop_size=config.VAD_HOP_SIZE,
        prepad_ms=config.VAD_PREPAD_MS,
        postpad_ms=config.VAD_POSTPAD_MS,
        min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
        min_silence_duration=config.VAD_MIN_SILENCE_DURATION,
    )
    GLOBAL_VAD_CORE = SileroVADCore(vad_cfg)
    CORE_LOCK = asyncio.Lock()
    logger.info("Startup complete.")


@app.get("/")
async def client_host():
    return FileResponse("realtime_ws_client_pcm.html", media_type="text/html")


@app.websocket("/api/realtime/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"Session {session_id} opened")

    if GLOBAL_CORE is None or GLOBAL_VAD_CORE is None or CORE_LOCK is None:
        await ws.close(code=1011)
        return

    session_asr = FunASRSession(GLOBAL_CORE)
    vad_cfg = VADConfig(
        model_path=config.VAD_MODEL,
        device=config.DEVICE,
        sample_rate=config.SAMPLERATE,
        threshold=config.VAD_THRESHOLD,
        window_size=config.VAD_WINDOW_SIZE,
        hop_size=config.VAD_HOP_SIZE,
        prepad_ms=config.VAD_PREPAD_MS,
        postpad_ms=config.VAD_POSTPAD_MS,
        min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
        min_silence_duration=config.VAD_MIN_SILENCE_DURATION,
    )
    session_vad = VADStream(GLOBAL_VAD_CORE, vad_cfg)

    chunk_size = int(config.CHUNK_DURATION * config.SAMPLERATE)
    audio_buf = np.zeros((0,), dtype=np.float32)

    speech_id = 0
    begin_at = 0.0
    in_speech = False
    last_text = ""
    first_result_sent_for_current_speech = False  # Track if first result has been sent for current speech segment

    async def handle_speech_audio(speech_f32: np.ndarray, is_last: bool, s0: int, s1: int):
        nonlocal last_text, speech_id, in_speech, begin_at, first_result_sent_for_current_speech
        if speech_f32.size == 0 and not is_last:
            return
        speech_i16 = np.clip(speech_f32 * 32768.0, -32768, 32767).astype(np.int16)

        # Collect all results during streaming
        collected_results = []
        
        async with CORE_LOCK:
            for res in session_asr.streaming_inference(speech_i16, is_last=bool(is_last)):
                collected_results.append(res)

        if collected_results:
            # Send first result immediately (only once per speech segment) to show first character
            if not first_result_sent_for_current_speech:
                first_res = collected_results[0]
                text = first_res.get("text", "") or ""
                if text:  # Only send if there's actual text
                    first_result_sent_for_current_speech = True
                    resp = TranscriptionResponse(
                        id=speech_id,
                        begin_at=begin_at,
                        end_at=None,
                        data=TranscriptionChunk(
                            timestamps=first_res.get("timestamps", []),
                            raw_text=text,
                            final_text=None,
                            delta=text,
                        ),
                        is_final=False,  # Not final yet
                        session_id=session_id,
                    )
                    await ws.send_json(resp.model_dump())
            
            # Only send final result when VAD ends (is_last=True) to avoid UI flickering
            # This ensures complete sentences are displayed together
            if is_last:
                final_res = collected_results[-1]
                text = final_res.get("text", "") or ""
                delta = final_res.get("delta", "") or ""
                final_text = final_res.get("final_text", None)

                display = final_text or text
                if display:
                    last_text = display

                resp = TranscriptionResponse(
                    id=speech_id,
                    begin_at=begin_at,
                    end_at=None,
                    data=TranscriptionChunk(
                        timestamps=final_res.get("timestamps", []),
                        raw_text=display,
                        final_text=final_text,
                        delta=delta,
                    ),
                    is_final=True,  # Always final when VAD ends
                    session_id=session_id,
                )
                await ws.send_json(resp.model_dump())
                # Reset flag for next speech segment
                first_result_sent_for_current_speech = False

        if is_last:
            end_at = float(s1) / config.SAMPLERATE
            resp2 = TranscriptionResponse(
                id=speech_id,
                begin_at=begin_at,
                end_at=end_at,
                data=TranscriptionChunk(raw_text=last_text, final_text=None, delta="", timestamps=[]),
                is_final=True,
                session_id=session_id,
            )
            await ws.send_json(resp2.model_dump())
            speech_id += 1
            in_speech = False
            last_text = ""
            await ws.send_json(VADEvent(is_active=False).model_dump())

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if msg["type"] == "websocket.receive" and "text" in msg:
                try:
                    j = json.loads(msg["text"])
                except Exception:
                    continue
                if isinstance(j, dict) and j.get("type") == "EOS":
                    logger.info(f"Session {session_id}: EOS received")
                    forced = session_vad.force_end()
                    if forced is not None:
                        speech_dict, speech_f32, is_last, s0, s1 = forced
                        await handle_speech_audio(speech_f32, True, s0, s1)
                    break
                continue

            if msg["type"] != "websocket.receive" or "bytes" not in msg:
                continue

            data = msg["bytes"]
            if not data:
                continue

            samples_i16 = np.frombuffer(data, dtype=np.int16)
            if samples_i16.size == 0:
                continue

            audio_max = int(np.abs(samples_i16).max())
            audio_mean = float(np.abs(samples_i16).mean())
            # Only warn if audio is suspiciously small (likely silence or wrong format)
            # Normal audio can have max < 200, so we use a lower threshold and check both max and mean
            if audio_max < 50 and audio_mean < 10:
                logger.debug(f"Audio seems very quiet: max={audio_max}, mean={audio_mean:.1f}")

            samples_f32 = samples_i16.astype(np.float32) / 32768.0
            audio_buf = np.concatenate([audio_buf, samples_f32], axis=0)

            while audio_buf.shape[0] >= chunk_size:
                chunk = audio_buf[:chunk_size]
                audio_buf = audio_buf[chunk_size:]

                events = session_vad.feed(chunk)
                for speech_dict, speech_f32, is_last, s0, s1 in events:
                    if "start" in speech_dict:
                        session_asr.reset()
                        begin_at = float(speech_dict["start"]) / config.SAMPLERATE
                        in_speech = True
                        first_result_sent_for_current_speech = False  # Reset flag for new speech segment
                        await ws.send_json(VADEvent(is_active=True).model_dump())

                    if in_speech or speech_f32.size > 0:
                        await handle_speech_audio(speech_f32, bool(is_last), s0, s1)

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    finally:
        logger.info(f"Session {session_id} closed")
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
