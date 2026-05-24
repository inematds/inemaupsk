"""inemaupsk — single-purpose image upscaling service.

Loads N models lazily (first use), caches them in VRAM. Endpoints:
  GET  /              -> playground UI
  GET  /health        -> liveness + which models are cached
  GET  /models        -> available models (filenames in /app/models/upscale)
  POST /upscale       -> {image, model?} -> {image, ...}
  POST /upscale_all   -> {image}         -> {results: [{model, image, elapsed_s}, ...]}

One asyncio lock serializes GPU access (shared GPU with inemaimg, no MPS).
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inemaupsk")

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from loaders.upscale import UpscaleLoader  # noqa: E402

MAX_INPUT_SIDE = 4096
MODELS_DIR = Path(os.environ.get("INEMAUPSK_MODELS_DIR", "/app/models/upscale"))
DEFAULT_MODEL = os.environ.get("INEMAUPSK_DEFAULT_MODEL", "4x-UltraSharp.pth")
WEB_INDEX = Path(__file__).parent / "web" / "index.html"


class _State:
    cache: dict[str, Any] = {}
    lock: asyncio.Lock | None = None


state = _State()


def _list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.pth"))


def _resolve_model(name: str | None) -> str:
    name = (name or DEFAULT_MODEL).strip()
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid model name")
    if not name.endswith(".pth"):
        name = name + ".pth"
    if not (MODELS_DIR / name).exists():
        raise HTTPException(
            status_code=404,
            detail=f"model {name} not found in {MODELS_DIR}",
        )
    return name


async def _get_model(name: str) -> Any:
    """Lazy-load + cache. Caller MUST hold state.lock for inference."""
    if name in state.cache:
        return state.cache[name]
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, UpscaleLoader.load, str(MODELS_DIR / name))
    state.cache[name] = model
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.lock = asyncio.Lock()
    log.info("models dir: %s, found: %s", MODELS_DIR, _list_models())
    yield


app = FastAPI(title="inemaupsk", version="0.2.0", lifespan=lifespan)


class UpscaleRequest(BaseModel):
    image: str = Field(..., description="base64-encoded PNG/JPEG")
    model: str | None = None


class UpscaleAllRequest(BaseModel):
    image: str


def _decode(b64: str) -> Image.Image:
    try:
        if "," in b64[:64]:  # data URL prefix
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {e}")


def _encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _check_size(img: Image.Image) -> None:
    long_side = max(img.size)
    if long_side > MAX_INPUT_SIDE:
        raise HTTPException(
            status_code=413,
            detail=f"input long side {long_side}px exceeds limit {MAX_INPUT_SIDE}px",
        )


@app.get("/", include_in_schema=False)
def index():
    if not WEB_INDEX.exists():
        raise HTTPException(status_code=404, detail="UI not bundled")
    return FileResponse(WEB_INDEX)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available": _list_models(),
        "cached": sorted(state.cache.keys()),
        "default": DEFAULT_MODEL,
        "max_input_side": MAX_INPUT_SIDE,
        "gpu_alloc_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        if torch.cuda.is_available() else 0.0,
    }


@app.get("/models")
def models():
    return {"models": _list_models(), "default": DEFAULT_MODEL}


@app.post("/upscale")
async def upscale(req: UpscaleRequest):
    name = _resolve_model(req.model)
    img = _decode(req.image)
    _check_size(img)
    assert state.lock is not None
    async with state.lock:
        model = await _get_model(name)
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        sr = await loop.run_in_executor(None, UpscaleLoader.upscale, model, img)
        elapsed = time.perf_counter() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("upscale[%s] %s -> %s in %.2fs", name, img.size, sr.size, elapsed)
    return {
        "model": name,
        "image": _encode(sr),
        "input_size": list(img.size),
        "output_size": list(sr.size),
        "elapsed_s": round(elapsed, 3),
    }


@app.post("/upscale_all")
async def upscale_all(req: UpscaleAllRequest):
    img = _decode(req.image)
    _check_size(img)
    names = _list_models()
    if not names:
        raise HTTPException(status_code=500, detail="no models installed")
    results = []
    assert state.lock is not None
    async with state.lock:
        loop = asyncio.get_running_loop()
        for name in names:
            try:
                model = await _get_model(name)
                t0 = time.perf_counter()
                sr = await loop.run_in_executor(None, UpscaleLoader.upscale, model, img)
                elapsed = time.perf_counter() - t0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results.append({
                    "model": name,
                    "image": _encode(sr),
                    "output_size": list(sr.size),
                    "elapsed_s": round(elapsed, 3),
                })
                log.info("upscale_all[%s] %.2fs", name, elapsed)
            except Exception as e:
                log.exception("upscale_all[%s] failed", name)
                results.append({"model": name, "error": f"{type(e).__name__}: {e}"})
    return {"input_size": list(img.size), "results": results}
