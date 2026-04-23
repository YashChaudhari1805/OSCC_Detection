"""
api/routes.py
-------------
All Flask endpoints for the OSCC Detection API.
"""

import base64
from flask import Blueprint, request, jsonify, current_app

from core.config import settings
from core.model import get_model, is_loaded, preprocess_image, run_inference
from core.security import protected, rate_limited
from core.xai import compute_gradcam, compute_lime

api_bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Auth ──────────────────────────────────────────────────────────────────────

@api_bp.route("/auth/verify", methods=["POST"])
@rate_limited
def verify_key():
    """
    Verify an API key. Used by the dashboard login screen.
    Returns auth_required flag so the UI knows whether to show the gate.
    """
    auth_required = settings.auth_enabled
    if not auth_required:
        return jsonify({"valid": True, "auth_required": False})

    key = request.headers.get("X-API-Key", "").strip()
    if not key:
        return jsonify({"valid": False, "auth_required": True})

    if key not in settings.api_keys:
        return jsonify({"error": "Invalid API key."}), 403

    return jsonify({"valid": True, "auth_required": True})


# ── Health ────────────────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": is_loaded(),
        "model_path": settings.model_path,
        "auth_enabled": settings.auth_enabled,
        "version": "1.0.0",
    })


# ── Predict ───────────────────────────────────────────────────────────────────

@api_bp.route("/predict", methods=["POST"])
@protected
def predict():
    """
    POST /api/predict
    Accepts multipart/form-data with a 'file' field.
    Returns prediction + Grad-CAM + LIME as base64 PNGs.
    """
    if not is_loaded():
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: PNG, JPG, JPEG, BMP, TIF."}), 400

    try:
        img_array, original_pil = preprocess_image(file)
        model = get_model()

        result = run_inference(img_array)
        gradcam_b64 = compute_gradcam(model, img_array, original_pil)
        lime_b64 = compute_lime(model, img_array, original_pil)

        file.seek(0)
        img_b64 = base64.b64encode(file.read()).decode("utf-8")

        return jsonify({
            "success": True,
            "prediction": result,
            "image": img_b64,
            "gradcam": gradcam_b64,
            "lime": lime_b64,
        })

    except Exception as e:
        current_app.logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Model info ────────────────────────────────────────────────────────────────

@api_bp.route("/model/info", methods=["GET"])
@protected
def model_info():
    """Return model architecture summary."""
    if not is_loaded():
        return jsonify({"error": "Model not loaded."}), 500

    model = get_model()
    layers = []
    for i, layer in enumerate(model.layers):
        layers.append({
            "index": i,
            "name": layer.name,
            "type": type(layer).__name__,
            "params": layer.count_params(),
        })

    return jsonify({
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "total_params": model.count_params(),
        "layers": layers,
    })
