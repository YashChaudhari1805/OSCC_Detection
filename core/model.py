"""
core/model.py
-------------
Model loading, image preprocessing, and inference.
"""

import os
import numpy as np
from PIL import Image

import keras
from keras.preprocessing import image as keras_image
from keras.applications.efficientnet import preprocess_input

from core.config import settings

_model = None


def load_model() -> bool:
    global _model
    if not os.path.exists(settings.model_path):
        print(f"[ERROR] Model not found at '{settings.model_path}'")
        return False
    try:
        print(f"[INFO] Loading model from '{settings.model_path}'...")
        _model = keras.models.load_model(settings.model_path, compile=False)
        _model.compile(
            keras.optimizers.Adamax(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f"[INFO] Model ready. Input: {_model.input_shape}")
        return True
    except Exception as e:
        print(f"[FATAL] Model load failed: {e}")
        return False


def get_model():
    return _model


def is_loaded() -> bool:
    return _model is not None


CLASS_LABELS = ["Normal", "OSCC"]
TARGET_SIZE = (settings.image_size, settings.image_size)


def preprocess_image(img_source) -> tuple[np.ndarray, Image.Image]:
    if isinstance(img_source, Image.Image):
        img = img_source
    else:
        img = Image.open(img_source)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr, img


def run_inference(img_array: np.ndarray) -> dict:
    scores = _model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(scores))
    return {
        "class": CLASS_LABELS[idx],
        "confidence": round(float(scores[idx]) * 100, 2),
        "scores": {
            CLASS_LABELS[i]: round(float(scores[i]) * 100, 2)
            for i in range(len(CLASS_LABELS))
        },
    }
