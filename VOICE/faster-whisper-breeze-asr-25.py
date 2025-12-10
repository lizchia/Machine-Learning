#!/usr/bin/env python3
import asyncio
import os
import wave
import math
import json
import numpy as np
import websockets
import time
from faster_whisper import WhisperModel

print("Script started...", flush=True)

###############################################################################
# Configuration (fixed codec + defaults for tunables)
###############################################################################
WAV_PATH = "capture.wav"
TXT_PATH = "transcript.txt"
HOST, PORT = "0.0.0.0", 8000
model_id = "SoybeanMilk/faster-whisper-Breeze-ASR-25"

SR = 16000
"""
Fixed
Target sample rate in Hz for the audio the server expects from the client. Must match your client-side resample. Whisper models work well at 16 kHz.
"""

FRAME_WIDTH_BYTES = 2
"""
Fixed
Bytes per sample. 16-bit PCM uses 2 bytes. If you ever switch to 24-bit or float, this must change, and your client must match.
"""

# Default tunables. These can be overridden per-session by the HTML via {"type":"config"}
DEFAULT_CONFIG = {
    "PAUSE_RMS": 0.005,        # treat below this as silence
    "PAUSE_MS": 1500,          # wall-clock pause to flush
    "MIN_SPOKEN_MS": 0,        # min speech since last flush
    "MAX_LATENCY_MS": 2500,    # force flush if tail waits too long
    "CHUNK_SECONDS": 5.0,      # nominal chunk size
    "OVERLAP_SECONDS": 0.5,    # reuse this much tail as context
}

###############################################################################
# Load faster-whisper model
###############################################################################
print(f"[init] Loading faster-whisper model '{model_id}'")
# Example CPU tiny:
# model = WhisperModel("tiny", device="cpu", compute_type="int8")
# Your current GPU setup:
model = WhisperModel(model_id, device="cuda", device_index=3, compute_type="int8")
print("[init] Model loaded.")

###############################################################################
# Utils
###############################################################################
def int16_bytes_to_float32(b: bytes) -> np.ndarray:
    a = np.frombuffer(b, dtype=np.int16)
    return a.astype(np.float32) / 32768.0

def append_text(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

def rms_float32(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))

def _coerce_config(params: dict) -> dict:
    """
    Defensive coercion/validation for incoming config.
    Only accepts known keys, correct types, basic sanity.
    """
    cfg = dict(DEFAULT_CONFIG)
    if not isinstance(params, dict):
        return cfg
    for k in cfg.keys():
        if k in params:
            try:
                if k in ("PAUSE_RMS", "CHUNK_SECONDS", "OVERLAP_SECONDS"):
                    v = float(params[k])
                else:
                    v = int(params[k])  # *_MS as ints
                # Basic sanity checks
                if k == "PAUSE_RMS" and v < 0: continue
                if k.endswith("_MS") and v < 0: continue
                if k == "CHUNK_SECONDS" and v < 1: continue
                if k == "OVERLAP_SECONDS" and v < 0: continue
                cfg[k] = v
            except Exception:
                pass
    # Ensure overlap <= chunk
    if cfg["OVERLAP_SECONDS"] > cfg["CHUNK_SECONDS"]:
        cfg["OVERLAP_SECONDS"] = float(cfg["CHUNK_SECONDS"])
    return cfg

###############################################################################
# Transcription worker (now takes a per-session cfg)
###############################################################################
async def transcribe_worker(audio_q: asyncio.Queue, stop_event: asyncio.Event, out_q: asyncio.Queue, cfg: dict):
    print("[transcribe_worker] started", flush=True)
    try:
        os.remove(TXT_PATH)
    except FileNotFoundError:
        pass

    # Pull local copies from cfg once here
    PAUSE_RMS        = float(cfg["PAUSE_RMS"])
    PAUSE_MS         = int(cfg["PAUSE_MS"])
    MIN_SPOKEN_MS    = int(cfg["MIN_SPOKEN_MS"])
    MAX_LATENCY_MS   = int(cfg["MAX_LATENCY_MS"])
    CHUNK_SECONDS    = float(cfg["CHUNK_SECONDS"])
    OVERLAP_SECONDS  = float(cfg["OVERLAP_SECONDS"])

    buffer = bytearray()
    processed_samples = 0
    last_emitted_end = 0.0
    silence_ms = 0.0
    spoken_ms_since_last_process = 0.0
    samples_per_chunk = int(SR * CHUNK_SECONDS)
    samples_overlap = int(SR * OVERLAP_SECONDS)
    chunk_index = 0

    # wall-clock tracking for pauses and latency cap
    last_speech_like_ts = time.monotonic()
    first_unprocessed_ts = None

    while True:
        try:
            b = await asyncio.wait_for(audio_q.get(), timeout=0.1)
            buffer.extend(b)
            audio_q.task_done()
        except asyncio.TimeoutError:
            pass

        total_samples = len(buffer) // FRAME_WIDTH_BYTES

        # mark when unprocessed audio starts
        if first_unprocessed_ts is None and total_samples > processed_samples:
            first_unprocessed_ts = time.monotonic()

        # --- Pause detector on the newest ~100ms window ---
        WINDOW_MS = 100
        win_samples = int(SR * (WINDOW_MS / 1000.0))
        tail_samples = total_samples - processed_samples
        if tail_samples > 0:
            take = min(tail_samples, win_samples)
            byte_start = (len(buffer) // FRAME_WIDTH_BYTES - take) * FRAME_WIDTH_BYTES
            tail_slice = int16_bytes_to_float32(buffer[byte_start: byte_start + take * FRAME_WIDTH_BYTES])
            r = rms_float32(tail_slice)
            if r < PAUSE_RMS:
                silence_ms += (take / SR) * 1000.0
                # do not reset last_speech_like_ts here
            else:
                silence_ms = 0.0
                spoken_ms_since_last_process += (take / SR) * 1000.0
                last_speech_like_ts = time.monotonic()

        # --- Decide if we should run ASR (chunk, pause, or latency cap, or STOP) ---
        need_run = (total_samples - processed_samples) >= samples_per_chunk

        if not need_run:
            have_fresh = (total_samples - processed_samples) > 0
            elapsed_since_speech_ms = (time.monotonic() - last_speech_like_ts) * 1000.0
            if have_fresh and elapsed_since_speech_ms >= PAUSE_MS and spoken_ms_since_last_process >= MIN_SPOKEN_MS:
                need_run = True

        if not need_run and first_unprocessed_ts:
            if (time.monotonic() - first_unprocessed_ts) * 1000.0 >= MAX_LATENCY_MS:
                need_run = True

        if stop_event.is_set() and total_samples > processed_samples:
            need_run = True

        if not need_run:
            if stop_event.is_set() and audio_q.empty():
                break
            await asyncio.sleep(0)
            continue

        chunk_index += 1
        est_total_chunks = max(1, math.ceil(total_samples / samples_per_chunk))
        await out_q.put({"type": "status",
                         "stage": "processing",
                         "chunk": chunk_index,
                         "est_total": est_total_chunks,
                         "buffered_sec": round(total_samples / SR, 2)})

        start_idx = max(0, processed_samples - samples_overlap)
        end_idx = total_samples
        audio_slice = int16_bytes_to_float32(buffer[start_idx * FRAME_WIDTH_BYTES : end_idx * FRAME_WIDTH_BYTES])
        t0 = start_idx / SR

        try:
            # built-in VAD to help segment on pauses
            segments, _ = await asyncio.to_thread(
                model.transcribe,
                audio_slice,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200
                ),
                # condition_on_previous_text=False,  # optional
            )
        except Exception as e:
            print(f"[transcribe_worker] ERROR on chunk {chunk_index}: {e}", flush=True)
            await out_q.put({"type": "error",
                             "chunk": chunk_index,
                             "message": f"transcribe failed: {e}"})
            processed_samples = end_idx
            spoken_ms_since_last_process = 0.0
            silence_ms = 0.0
            last_speech_like_ts = time.monotonic()
            first_unprocessed_ts = None
            continue

        new_texts = []
        for seg in segments:
            s_global = seg.start + t0
            e_global = seg.end   + t0
            txt = (seg.text or "").strip()
            print(f"[chunk {chunk_index}] text='{txt}'", flush=True)
            # accept if segment extends past watermark
            if e_global > last_emitted_end + 0.05:
                new_texts.append(txt)
                last_emitted_end = max(last_emitted_end, e_global)

        if new_texts:
            out_line = " ".join(new_texts).strip()
            if out_line:
                print(f"[chunk {chunk_index}] {out_line}", flush=True)
                append_text(TXT_PATH, out_line + "\n")
                await out_q.put({"type": "partial", "chunk": chunk_index, "text": out_line})
        else:
            print(f"[chunk {chunk_index}] (no new text)", flush=True)
            await out_q.put({"type": "partial", "chunk": chunk_index, "text": ""})

        await out_q.put({"type": "status", "stage": "done", "chunk": chunk_index})

        processed_samples = end_idx
        # reset counters after a run
        spoken_ms_since_last_process = 0.0
        silence_ms = 0.0
        last_speech_like_ts = time.monotonic()
        first_unprocessed_ts = None

        if stop_event.is_set() and audio_q.empty():
            break

    await out_q.put({"type": "status", "stage": "worker_done"})
    print("[transcribe_worker] done", flush=True)

###############################################################################
# WebSocket handler (accepts config, starts worker lazily)
###############################################################################
async def handler(ws):
    print("[handler] client connected")

    # Per-connection state
    cfg = dict(DEFAULT_CONFIG)   # active config for this client
    stop_event = asyncio.Event()
    audio_q = asyncio.Queue()
    out_q = asyncio.Queue()
    worker_task = None           # start lazily after config or first audio

    # Prepare WAV writer immediately (codec is fixed)
    wf = wave.open(WAV_PATH, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    frames_written = 0

    async def ensure_worker_started():
        nonlocal worker_task
        if worker_task is None:
            # announce the config we are using
            await out_q.put({"type": "status", "stage": "config_active", "params": cfg})
            worker_task = asyncio.create_task(transcribe_worker(audio_q, stop_event, out_q, cfg))
            print(f"[handler] worker started with cfg={cfg}", flush=True)

    async def sender():
        # Forward JSON messages from out_q to the client
        try:
            while True:
                msg = await out_q.get()
                try:
                    await ws.send(json.dumps(msg, ensure_ascii=False))
                except websockets.exceptions.ConnectionClosed:
                    break
                finally:
                    out_q.task_done()
        except asyncio.CancelledError:
            pass

    sender_task = asyncio.create_task(sender())

    try:
        async for msg in ws:
            # Handle STOP and CONFIG control messages
            if isinstance(msg, str):
                m = msg.strip()
                if not m:
                    continue
                # Try JSON first
                try:
                    parsed = json.loads(m)
                except Exception:
                    parsed = None

                if isinstance(parsed, dict) and parsed.get("type") == "config":
                    # Apply new config for THIS connection, before worker starts
                    new_cfg = _coerce_config(parsed.get("params", {}))
                    cfg = new_cfg
                    print(f"[handler] received config: {cfg}", flush=True)
                    await out_q.put({"type": "status", "stage": "config_applied", "params": cfg})
                    # Do not start worker yet, wait for first audio or next step
                    continue

                # Plain STOP string
                if m.upper() == "STOP":
                    await out_q.put({"type": "status", "stage": "stop_received"})
                    break

                # Unknown text message, ignore
                continue

            # Binary audio
            if isinstance(msg, (bytes, bytearray)):
                # Start worker on demand if not started yet
                if worker_task is None:
                    await ensure_worker_started()
                wf.writeframes(msg)
                frames_written += len(msg) // FRAME_WIDTH_BYTES
                await audio_q.put(msg)
                if frames_written % (SR * 5) == 0:
                    await out_q.put({"type": "status",
                                     "stage": "audio_progress",
                                     "seconds": round(frames_written / SR, 2)})
    except websockets.exceptions.ConnectionClosed:
        print("[handler] connection closed")
    finally:
        # graceful shutdown
        stop_event.set()
        await audio_q.join()
        if worker_task is not None:
            await worker_task
        try:
            await out_q.join()
        except Exception:
            pass
        try:
            sender_task.cancel()
        except Exception:
            pass
        wf.close()
        print(f"[handler] client disconnected, saved ≈ {frames_written / SR:.2f}s audio")

###############################################################################
# Server main
###############################################################################
async def main():
    print(f"[server] listening on ws://{HOST}:{PORT}")
    async with websockets.serve(
        handler,
        HOST,
        PORT,
        max_size=None,
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())






