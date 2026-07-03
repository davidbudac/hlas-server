# CLAUDE.md

## Overview

Hlas Server is a speech-to-text transcription API using NVIDIA's Parakeet TDT 0.6B v3 model. It serves as the remote transcription backend for the [Hlas](https://github.com/davidbudac/hlas) macOS app.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI with Uvicorn
- **Model:** NVIDIA Parakeet TDT 0.6B v3 (via NeMo ASR)
- **Runtime:** Docker with NVIDIA CUDA 12.8.1

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns `{"status": "ok", "model_loaded": true/false}` |
| `/transcribe` | POST | Transcribe WAV audio, multipart form with `file` field |

**Audio requirements:** 16kHz sample rate, WAV format. Multi-channel is auto-converted to mono.

## Build & Run

```bash
# Build
docker build -t hlas-server .

# Run with GPU
docker run -d --name hlas --gpus all -p 8000:8000 \
  -v hlas-models:/home/appuser/.cache \
  --restart unless-stopped hlas-server
```

Requires NVIDIA Container Toolkit for `--gpus all`.

## Running on Apple Silicon (M-series Macs)

The CUDA/NeMo path above is NVIDIA-only and cannot use the Mac's Metal GPU (Docker
on macOS can't pass it through). For M-series Macs there is a parallel, native
server that exposes the **same** `/health` and `/transcribe` API, so the Hlas app
can point at either backend unchanged.

- **`server_mac.py`** — uses [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx)
  (MLX / Metal) instead of NeMo/CUDA. Runs the same model via the Apple-Silicon
  conversion `mlx-community/parakeet-tdt-0.6b-v3`.
- **`requirements-mac.txt`** — lean deps (no torch, no nemo_toolkit).

```bash
pip install -r requirements-mac.txt
uvicorn server_mac:app --host 0.0.0.0 --port 8000
```

Run it **natively, not in Docker** (Docker on macOS can't reach the Metal GPU).
First launch downloads the model (~600 MB) into `~/.cache/huggingface` and caches
it thereafter. Transcription runs faster than real-time on an M4.

## Deployment

Runs on a local GPU server (RTX 4070, 12GB VRAM). Model cache persists in the `hlas-models` Docker volume.
