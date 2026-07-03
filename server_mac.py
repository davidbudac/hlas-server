"""Apple Silicon (M-series) transcription server.

Mirrors the API of server.py (the CUDA/NeMo server) so the Hlas macOS app can
point at either backend, but runs the same Parakeet TDT 0.6B v3 model natively
on the Mac's Metal GPU via MLX (https://github.com/senstella/parakeet-mlx)
instead of NeMo/CUDA.

Run:
    pip install -r requirements-mac.txt
    uvicorn server_mac:app --host 0.0.0.0 --port 8000
"""

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from parakeet_mlx import from_pretrained

# MLX community mirror of nvidia/parakeet-tdt-0.6b-v3, converted for Apple Silicon.
MODEL_ID = os.environ.get("HLAS_MODEL_ID", "mlx-community/parakeet-tdt-0.6b-v3")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading {MODEL_ID} (MLX) ...")
    model = from_pretrained(MODEL_ID)
    print("Model loaded and ready.")
    yield


app = FastAPI(title="Hlas Parakeet Server (MLX)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(contents), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    if sample_rate != 16000:
        raise HTTPException(
            status_code=400, detail=f"Expected 16kHz audio, got {sample_rate}Hz"
        )

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration = len(audio_data) / 16000
    print(f"[transcribe] Received {duration:.1f}s audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_data, 16000)
        tmp_path = tmp.name

    try:
        start = time.time()
        result = model.transcribe(tmp_path)
        elapsed = time.time() - start
        text = result.text
        print(f"[transcribe] '{text.strip()}' ({elapsed:.3f}s)")
    finally:
        os.unlink(tmp_path)

    return JSONResponse({"text": text.strip(), "duration_seconds": round(elapsed, 3)})
