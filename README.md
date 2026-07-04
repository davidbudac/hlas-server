# hlas-server

Remote speech-to-text transcription server for [Hlas](https://github.com/davidbudac/hlas), powered by NVIDIA's Parakeet TDT 0.6B v3 model.

## Quickstart

```bash
docker build -t hlas-server .
docker run -d --name hlas --gpus all -p 8000:8000 \
  -v hlas-models:/home/appuser/.cache \
  --restart unless-stopped hlas-server
```

Requires an NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## API

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

### `POST /transcribe`

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
# {"text": "hello world", "duration_seconds": 0.342}
```

Audio must be 16kHz WAV. Multi-channel files are automatically mixed to mono.

## Running without Docker

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

The model (~1.2 GB) downloads automatically on first startup.

## Running on Apple Silicon (M-series Macs)

The Docker/CUDA setup is NVIDIA-only. On M-series Macs, use the native MLX
server, which exposes the same `/health` and `/transcribe` API (so the Hlas app
needs no changes) but runs on the Metal GPU via
[`parakeet-mlx`](https://github.com/senstella/parakeet-mlx):

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-mac.txt
uv run --no-project --python .venv/bin/python \
  uvicorn server_mac:app --host 0.0.0.0 --port 8000
```

Run it natively, not in Docker (Docker on macOS can't reach the Metal GPU). The
model (~600 MB) downloads on first startup and is cached afterward.
Transcription runs faster than real-time on an M4.
