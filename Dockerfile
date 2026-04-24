# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install tensorflow-cpu first (largest package, ~214 MB) so it gets its own
# cached layer. Then install the rest. --retries 10 and --timeout 120 guard
# against the transient network drops seen in the original build log.
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

# Copy only the runtime files — NOT api/ (Flask remnant, unused by FastAPI app)
COPY core/       ./core/
COPY templates/  ./templates/
COPY static/     ./static/
COPY main.py     .

# Ensure static dir always exists (app.mount won't error on missing dir)
RUN mkdir -p static/css static/js

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["python", "main.py"]
