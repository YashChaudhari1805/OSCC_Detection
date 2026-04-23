"""tests/conftest.py — shared fixtures for OSCC v2 (FastAPI)"""
from __future__ import annotations
import io, sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session", autouse=True)
def mock_model():
    fake = MagicMock()
    fake.input_shape  = (None, 224, 224, 3)
    fake.output_shape = (None, 2)
    fake.predict.return_value = np.array([[0.08, 0.92]])
    fake.layers = []
    with patch("core.model._model", fake):
        yield fake

@pytest.fixture(scope="session")
def client(mock_model):
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

@pytest.fixture
def jpeg_bytes() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (8, 8), (180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
