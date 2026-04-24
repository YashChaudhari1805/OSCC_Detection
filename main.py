"""
main.py
-------
OSCC Detection — FastAPI application entry point.
Run: python main.py  OR  uvicorn main:app --reload
"""

import base64
import io
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

load_dotenv()

from core.config import settings
from core.model import get_model, is_loaded, load_model, preprocess_image, run_inference
from core.xai import compute_gradcam, compute_lime

# ── Lifespan (load model on startup) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 55)
    print("  OSCC Detection — FastAPI Server")
    print("=" * 55)
    ok = load_model()
    if not ok:
        print("[WARN] Model not loaded. /api/predict will return 503.")
        print("  Run: python rebuild_model.py  (needs my_final_oscc_model.h5)")
    else:
        print(f"[OK] Serving on http://{settings.host}:{settings.port}")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OSCC Detection API",
    description="Histopathological image classifier — EfficientNetB3 + Grad-CAM + LIME",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Serve static assets — only mount if the folder exists AND has content,
# so a fresh Docker image (no static files) doesn't raise a startup error.
_STATIC = Path(__file__).parent / "static"
if _STATIC.exists() and any(_STATIC.iterdir()):
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ── Rate limiter ──────────────────────────────────────────────────────────────

_windows: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    w = _windows[ip]
    now = time.monotonic()
    cutoff = now - 60.0
    while w and w[0] < cutoff:
        w.popleft()
    if len(w) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit: max {settings.rate_limit_per_minute} req/min.",
        )
    w.append(now)


# ── API key auth ──────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_key(x_api_key: Optional[str] = Depends(_api_key_header)) -> None:
    if not settings.auth_enabled:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    if x_api_key not in settings.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key.")


_protected = [Depends(rate_limit), Depends(verify_key)]


# ── File validation ───────────────────────────────────────────────────────────

_ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def _check_file(file: UploadFile) -> None:
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if not file.filename or ext not in _ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type. Allowed: {', '.join(sorted(_ALLOWED)).upper()}.",
        )
    if file.size and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_upload_mb} MB.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def dashboard():
    p = Path(__file__).parent / "templates" / "index.html"
    if not p.exists():
        return JSONResponse({"error": "Dashboard not found."}, status_code=404)
    return FileResponse(str(p))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "model_path": settings.model_path,
        "auth_enabled": settings.auth_enabled,
        "version": "2.0.0",
    }


@app.post("/api/auth/verify")
def auth_verify(
    request: Request,
    x_api_key: Optional[str] = Depends(_api_key_header),
):
    rate_limit(request)
    if not settings.auth_enabled:
        return {"valid": True, "auth_required": False}
    if not x_api_key:
        return {"valid": False, "auth_required": True}
    if x_api_key not in settings.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return {"valid": True, "auth_required": True}


@app.post("/api/predict", dependencies=_protected)
async def predict(file: UploadFile = File(...)):
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run rebuild_model.py first.",
        )

    _check_file(file)

    raw = await file.read()
    # Secondary size check (file.size may be None for streamed uploads)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_upload_mb} MB.",
        )

    try:
        img_array, pil_img = preprocess_image(io.BytesIO(raw))
        model = get_model()
        result   = run_inference(img_array)
        gradcam  = compute_gradcam(model, img_array, pil_img)
        lime_out = compute_lime(model, img_array, pil_img)
        img_b64  = base64.b64encode(raw).decode()

        return {
            "success": True,
            "prediction": result,
            "image": img_b64,
            "gradcam": gradcam,
            "lime": lime_out,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level,
    )
