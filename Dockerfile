# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
        --retries 10 --timeout 120 \
        tensorflow-cpu==2.16.1

RUN pip install --no-cache-dir --prefix=/install \
        --retries 10 --timeout 120 \
        -r requirements.txt

# ── Stage 2: slim runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY core/       ./core/
COPY templates/  ./templates/
COPY static/     ./static/
COPY main.py     .

# Copy the model file (must be real file, not LFS pointer)
# If using Git LFS, Render must have LFS enabled or model committed directly
COPY oscc_model_rebuilt.keras .

RUN mkdir -p static/css static/js

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]