# CLAUDE.md

## Overview

Hlas Server is a speech-to-text transcription API using NVIDIA's Parakeet TDT 0.6B v2 model. It serves as the remote transcription backend for the [Hlas](https://github.com/davidbudac/hlas) macOS app.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI with Uvicorn
- **Model:** NVIDIA Parakeet TDT 0.6B v2 (via NeMo ASR)
- **Runtime:** Docker with NVIDIA CUDA 12.4.1

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

## Deployment

Runs on a local GPU server (RTX 4070, 12GB VRAM). Model cache persists in the `hlas-models` Docker volume.
