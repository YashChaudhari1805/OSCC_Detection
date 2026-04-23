"""tests/test_api.py — FastAPI integration tests"""
from __future__ import annotations
import io
from unittest.mock import patch
import numpy as np
import pytest


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert "model_loaded" in d
    assert "version" in d


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"OSCC" in r.content


def test_auth_verify_no_auth(client):
    """When API_KEYS is blank, verify returns auth_required=False."""
    with patch("main.settings") as ms:
        ms.auth_enabled = False
        ms.rate_limit_per_minute = 60
        r = client.post("/api/auth/verify")
    assert r.status_code == 200
    assert r.json()["auth_required"] is False


def test_predict_no_file(client):
    r = client.post("/api/predict")
    assert r.status_code == 422  # FastAPI validation error


def test_predict_bad_extension(client, jpeg_bytes):
    r = client.post(
        "/api/predict",
        files={"file": ("slide.pdf", io.BytesIO(jpeg_bytes), "application/pdf")},
    )
    assert r.status_code == 400


def test_predict_success(client, jpeg_bytes):
    from PIL import Image
    fake_pil = Image.new("RGB", (224, 224))
    fake_arr = np.zeros((1, 224, 224, 3), dtype="float32")
    fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQ==AAF"

    with (
        patch("main.preprocess_image", return_value=(fake_arr, fake_pil)),
        patch("main.run_inference", return_value={
            "class": "OSCC", "confidence": 92.0,
            "scores": {"Normal": 8.0, "OSCC": 92.0},
        }),
        patch("main.compute_gradcam", return_value=fake_b64),
        patch("main.compute_lime",    return_value=fake_b64),
    ):
        r = client.post(
            "/api/predict",
            files={"file": ("slide.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        )

    assert r.status_code == 200
    d = r.json()
    assert d["success"] is True
    assert d["prediction"]["class"] == "OSCC"
    assert "gradcam" in d
    assert "lime" in d


def test_404_returns_json(client):
    r = client.get("/nonexistent")
    assert r.status_code == 404
