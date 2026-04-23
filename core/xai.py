"""
core/xai.py
-----------
Explainable AI: Grad-CAM and LIME implementations.

Uses native keras 3 (import keras) and tensorflow for GradientTape only.
No tensorflow.keras or tf_keras imports — both are broken on Windows
with the tensorflow-intel distribution.
"""

import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

import keras
from keras.applications.efficientnet import preprocess_input as eff_preprocess

from core.config import settings

_lime_explainer = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(model, img_array: np.ndarray, original_pil: Image.Image) -> str:
    """Compute Grad-CAM heatmap and return base64-encoded PNG."""
    base_model = model.get_layer("efficientnetb3")

    # Build sub-model using keras 3 Model class directly
    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("top_conv").output,
            base_model.output,
        ],
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, pooled_output = feature_extractor(img_tensor, training=False)
        tape.watch(conv_outputs)
        # Run the Sequential head layers manually (everything after base_model)
        x = pooled_output
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        pred_index = tf.argmax(x[0])
        class_channel = x[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    img_np = np.array(original_pil)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0d0f14")

    panels = [
        ("Original", img_np,         None),
        ("Heatmap",  heatmap_resized, "jet"),
        ("Overlay",  superimposed,    None),
    ]
    for ax, (title, im, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10, color="#c8cdd8", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig)


# ── LIME ──────────────────────────────────────────────────────────────────────

def compute_lime(model, img_array: np.ndarray, original_pil: Image.Image) -> str:
    """Compute LIME superpixel explanation and return base64-encoded PNG."""
    global _lime_explainer
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    if _lime_explainer is None:
        _lime_explainer = lime_image.LimeImageExplainer()

    def predict_fn(images: np.ndarray) -> np.ndarray:
        # images: float64 numpy array from LIME, shape (N, H, W, 3)
        images = eff_preprocess(images.astype("float32"))
        return model.predict(images, verbose=0)

    img_np = np.array(original_pil)

    explanation = _lime_explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=settings.lime_num_samples,
        random_seed=42,
    )

    top_label = explanation.top_labels[0]
    temp_pos, mask_pos = explanation.get_image_and_mask(
        top_label, positive_only=True,
        num_features=settings.lime_num_features, hide_rest=False,
    )
    temp_both, mask_both = explanation.get_image_and_mask(
        top_label, positive_only=False,
        num_features=settings.lime_num_features, hide_rest=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0d0f14")

    panels = [
        ("Original",     img_np,    None),
        ("LIME Regions", temp_pos,  mask_pos),
        ("+ vs −",       temp_both, mask_both),
    ]
    for ax, (title, img, mask) in zip(axes, panels):
        if mask is not None:
            ax.imshow(mark_boundaries(img, mask))
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10, color="#c8cdd8", pad=8)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig)
