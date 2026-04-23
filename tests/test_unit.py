"""tests/test_unit.py — unit tests for core modules"""
from __future__ import annotations
import io
from unittest.mock import patch, MagicMock
import numpy as np
import pytest


class TestConfig:
    def test_empty_keys_disables_auth(self):
        from core.config import Settings
        s = Settings(api_keys="")
        assert s.auth_enabled is False
        assert s.api_keys == []

    def test_comma_separated_keys_parsed(self):
        from core.config import Settings
        s = Settings(api_keys="aaa, bbb , ccc")
        assert s.api_keys == ["aaa", "bbb", "ccc"]

    def test_auth_enabled_with_keys(self):
        from core.config import Settings
        assert Settings(api_keys="k1").auth_enabled is True

    def test_max_upload_bytes(self):
        from core.config import Settings
        assert Settings(max_upload_mb=8).max_upload_bytes == 8 * 1024 * 1024


class TestInference:
    def test_oscc_prediction(self):
        from core.model import run_inference
        fake = MagicMock()
        fake.predict.return_value = np.array([[0.06, 0.94]])
        with patch("core.model._model", fake):
            r = run_inference(np.zeros((1, 224, 224, 3)))
        assert r["class"] == "OSCC"
        assert r["confidence"] == 94.0

    def test_normal_prediction(self):
        from core.model import run_inference
        fake = MagicMock()
        fake.predict.return_value = np.array([[0.91, 0.09]])
        with patch("core.model._model", fake):
            r = run_inference(np.zeros((1, 224, 224, 3)))
        assert r["class"] == "Normal"
        assert r["confidence"] == 91.0

    def test_scores_sum_100(self):
        from core.model import run_inference
        fake = MagicMock()
        fake.predict.return_value = np.array([[0.35, 0.65]])
        with patch("core.model._model", fake):
            r = run_inference(np.zeros((1, 224, 224, 3)))
        assert abs(sum(r["scores"].values()) - 100.0) < 0.1


class TestPreprocessing:
    def test_output_shape(self):
        from PIL import Image
        from core.model import preprocess_image
        img = Image.new("RGB", (400, 300))
        buf = io.BytesIO(); img.save(buf, format="JPEG"); buf.seek(0)
        arr, pil = preprocess_image(buf)
        assert arr.shape == (1, 224, 224, 3)
        assert pil.size == (224, 224)

    def test_rgba_to_rgb(self):
        from PIL import Image
        from core.model import preprocess_image
        img = Image.new("RGBA", (64, 64), (100, 100, 100, 200))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        arr, _ = preprocess_image(buf)
        assert arr.shape[-1] == 3


class TestRateLimiter:
    def test_under_limit_passes(self):
        from core.config import Settings
        import main
        # patch the windows dict so we start fresh
        original = main._windows.copy()
        main._windows.clear()
        from fastapi.testclient import TestClient
        with TestClient(main.app) as c:
            r = c.get("/api/health")
        assert r.status_code == 200
        main._windows = original
