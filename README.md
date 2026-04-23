<div align="center">

# OSCC Detection · Clinical AI Dashboard

**Oral Squamous Cell Carcinoma detection from histopathological slides — EfficientNetB3 + Grad-CAM + LIME**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TF--CPU-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## Overview

A clinical-grade binary image classifier detecting **OSCC** from histopathological slides. Built on EfficientNetB3 transfer learning, served via a **FastAPI** backend, with **Grad-CAM** and **LIME** explainability baked in.

The dashboard is a single self-contained HTML file — glassmorphic dark/light UI, scan-line animation, working auth gate, show/hide password, responsive on mobile and desktop.

---

## Quick Start

```powershell
# 1 — Clone and create venv
git clone https://github.com/you/oscc-detection.git
cd oscc-detection
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2 — Install (tensorflow-cpu avoids the broken Windows intel stub)
pip install -r requirements.txt

# 3 — Rebuild model (needs my_final_oscc_model.h5 in the project root)
python rebuild_model.py

# 4 — Configure
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
# Edit .env — set API_KEYS if you want auth

# 5 — Run
python main.py
```

Open **http://localhost:8000** · API docs at **http://localhost:8000/docs**

---

## Why `tensorflow-cpu` not `tensorflow`?

On Windows, `pip install tensorflow` installs **`tensorflow-intel`** — a stub package that omits `tensorflow.python.trackable` and several other internal modules, breaking keras 3.  
`tensorflow-cpu` is the real build and works correctly on all platforms.

---

## Docker (Local)

```powershell
# Build the model file first (outside Docker)
python rebuild_model.py

copy .env.example .env   # fill in API_KEYS etc.
docker compose up --build
```

The `.keras` model is mounted read-only at runtime — it is never baked into the image.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | *(empty)* | Comma-separated keys. Empty = auth disabled. |
| `MODEL_PATH` | `oscc_model_rebuilt.keras` | Path to the `.keras` model file |
| `PORT` | `8000` | Server port |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per IP per minute |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `LOG_LEVEL` | `info` | Uvicorn log level |
| `LIME_NUM_SAMPLES` | `1000` | LIME perturbations (higher = slower but more accurate) |
| `MAX_UPLOAD_MB` | `16` | Max upload file size |

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | — | Dashboard UI |
| `GET` | `/api/health` | No | Health + model status |
| `POST` | `/api/auth/verify` | No | Verify API key (used by login gate) |
| `POST` | `/api/predict` | Yes | Classify + Grad-CAM + LIME |

Interactive Swagger docs: **http://localhost:8000/docs**

### Predict example

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "X-API-Key: your_key" \
  -F "file=@slide.jpg"
```

```json
{
  "success": true,
  "prediction": {
    "class": "OSCC",
    "confidence": 94.73,
    "scores": { "Normal": 5.27, "OSCC": 94.73 }
  },
  "image":   "<base64 JPEG>",
  "gradcam": "<base64 PNG>",
  "lime":    "<base64 PNG>"
}
```

---

## Project Structure

```
oscc-detection/
├── core/
│   ├── config.py        Pydantic-settings (all env vars)
│   ├── model.py         Load, preprocess, run inference
│   └── xai.py           Grad-CAM and LIME
├── templates/
│   └── index.html       Complete dashboard (CSS + JS fully inline — no load-order bugs)
├── tests/
│   ├── conftest.py      Shared fixtures
│   ├── test_api.py      FastAPI integration tests
│   └── test_unit.py     Config / inference / preprocessing / rate-limiter tests
├── main.py              FastAPI app + all routes + rate limiter + auth
├── rebuild_model.py     Rebuild .keras from original .h5 weights
├── Dockerfile           Multi-stage slim build
├── docker-compose.yml   Local deployment
├── render.yaml          Render Docker deployment
├── requirements.txt     Dependencies (tensorflow-cpu, fastapi, uvicorn…)
└── .env.example         Config template
```

---

## Deployment on Render

`render.yaml` uses `runtime: docker` — Render builds from your `Dockerfile` directly.

**The model file:** `.keras` is gitignored. For Render free tier, temporarily commit it:

```bash
# Remove *.keras from .gitignore temporarily
git add oscc_model_rebuilt.keras
git commit -m "add model for deploy"
git push
# Re-add to .gitignore afterward
```

**Steps:**
1. Push repo to GitHub
2. Render → New → Web Service → connect repo
3. Set env vars in Render dashboard: `API_KEYS`, `SECRET_KEY`, `ENVIRONMENT=production`, `ALLOWED_ORIGINS=https://your-app.onrender.com`
4. Deploy — Docker build takes ~5–8 min (TF is large)

---

## Running Tests

```bash
pytest -v
```

---

## Known Limitations

- Research/demonstration only — not a certified medical device
- LIME takes ~60–90 seconds (1000 perturbation samples)
- Magnification bias: performs better on 400× slides than 100×
- Free Render tier: no persistent disk, ~30s cold start after 15 min idle
