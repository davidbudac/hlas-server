import io
import logging
import os
import sys
import tempfile
import time
import warnings

# Suppress all warnings and noisy library logs before imports
os.environ["NEMO_NOLOGGING"] = "1"
os.environ["MEGATRON_LOG_LEVEL"] = "50"
warnings.filterwarnings("ignore")
for name in ["nemo_logger", "nemo.collections", "nemo.utils", "nemo.core",
             "pytorch_lightning", "lhotse", "megatron", "onelogger", "opentelemetry"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Silence stderr during imports (Megatron/OneLogger print directly)
_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import numpy as np
import soundfile as sf

# Patch lhotse compatibility with PyTorch 2.10+ (Sampler no longer accepts data_source)
import lhotse.dataset.sampling.base as _lhotse_base

_orig_init = _lhotse_base.CutSampler.__init__


def _patched_init(self, *args, **kwargs):
    try:
        _orig_init(self, *args, **kwargs)
    except TypeError:
        import torch.utils.data

        _orig_sampler_init = torch.utils.data.Sampler.__init__
        torch.utils.data.Sampler.__init__ = lambda self_inner, *a, **kw: None
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            torch.utils.data.Sampler.__init__ = _orig_sampler_init


_lhotse_base.CutSampler.__init__ = _patched_init

import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging
nemo_logging.setLevel(logging.ERROR)
import torch

# Restore stderr
sys.stderr = _stderr

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading nvidia/parakeet-tdt-0.6b-v3 ...")
    # Silence stderr during model load (NeMo prints training config warnings)
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    if hasattr(model, "decoding") and hasattr(model.decoding, "decoding"):
        model.decoding.decoding.disable_cuda_graphs()
    sys.stderr = _stderr
    print("Model loaded and ready.")
    yield


app = FastAPI(title="Hlas Parakeet Server", lifespan=lifespan)


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
        # Silence NeMo warnings during transcription
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        output = model.transcribe([tmp_path])
        sys.stderr = _stderr
        elapsed = time.time() - start

        if isinstance(output[0], str):
            text = output[0]
        else:
            text = output[0].text if hasattr(output[0], "text") else str(output[0])
        print(f"[transcribe] '{text.strip()}' ({elapsed:.3f}s)")
    finally:
        os.unlink(tmp_path)

    return JSONResponse({"text": text.strip(), "duration_seconds": round(elapsed, 3)})
