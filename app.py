from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adamax
import numpy as np
import base64
import os
import io
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required for Flask
import matplotlib.pyplot as plt
from PIL import Image

# ── LIME (lazy import — only loaded once on first predict) ─────────────────────
lime_explainer = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

MODEL_PATH = 'oscc_model_rebuilt.keras'
IMAGE_SIZE = (224, 224)
CLASS_LABELS = ['Normal', 'OSCC']  # Alphabetical — matches Keras directory scanning

model = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_app_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f" * ERROR: Model file not found at '{MODEL_PATH}'")
            return False
        print(f" * Loading model from '{MODEL_PATH}'...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f" * Model loaded. Input shape: {model.input_shape}")
        return True
    except Exception as e:
        print(f" * FATAL: Could not load model — {e}")
        return False


def preprocess_image(img_source, target_size=IMAGE_SIZE):
    """Accept a file-like object or PIL Image, return preprocessed numpy array + PIL Image."""
    if isinstance(img_source, Image.Image):
        img = img_source
    else:
        img = Image.open(img_source)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def get_prediction(img_array):
    scores = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(scores))
    return {
        'class': CLASS_LABELS[predicted_index],
        'confidence': round(float(scores[predicted_index]) * 100, 2),
        'scores': {CLASS_LABELS[i]: round(float(scores[i]) * 100, 2) for i in range(len(CLASS_LABELS))}
    }


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(img_array, original_pil):
    """Return base64-encoded Grad-CAM overlay image."""
    base_model = model.get_layer('efficientnetb3')

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("top_conv").output,
            base_model.output
        ]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, pooled_output = feature_extractor(img_tensor, training=False)
        tape.watch(conv_outputs)
        x = pooled_output
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img_np = np.array(original_pil)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#f5f2eb')
    titles = ['Original', 'Heatmap', 'Grad-CAM Overlay']
    images = [img_np, heatmap_resized, superimposed]
    cmaps = [None, 'jet', None]

    for ax, title, im, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10, color='#1a1814', pad=8)
        ax.axis('off')

    fig.tight_layout()
    return fig_to_base64(fig)


# ── LIME ──────────────────────────────────────────────────────────────────────

def compute_lime(img_array, original_pil):
    """Return base64-encoded LIME explanation image."""
    global lime_explainer
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    if lime_explainer is None:
        lime_explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = tf.cast(images, tf.float32)
        images = preprocess_input(images.numpy())
        return model.predict(images, verbose=0)

    img_np = np.array(original_pil)

    explanation = lime_explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000,
        random_seed=42
    )

    top_label = explanation.top_labels[0]

    temp_pos, mask_pos = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=10, hide_rest=False
    )
    temp_both, mask_both = explanation.get_image_and_mask(
        top_label, positive_only=False, num_features=10, hide_rest=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#f5f2eb')

    axes[0].imshow(img_np)
    axes[0].set_title('Original', fontsize=10, color='#1a1814', pad=8)
    axes[0].axis('off')

    axes[1].imshow(mark_boundaries(temp_pos, mask_pos))
    axes[1].set_title('LIME Regions', fontsize=10, color='#1a1814', pad=8)
    axes[1].axis('off')

    axes[2].imshow(mark_boundaries(temp_both, mask_both))
    axes[2].set_title('Positive (green) vs Negative (red)', fontsize=10, color='#1a1814', pad=8)
    axes[2].axis('off')

    fig.tight_layout()
    return fig_to_base64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Allowed: PNG, JPG, JPEG, BMP, TIF.'}), 400

    try:
        img_array, original_pil = preprocess_image(file)
        result = get_prediction(img_array)

        # Both XAI methods run automatically on every prediction
        gradcam_b64 = compute_gradcam(img_array, original_pil)
        lime_b64 = compute_lime(img_array, original_pil)

        # Encode original image for display
        file.seek(0)
        img_data = base64.b64encode(file.read()).decode('utf-8')

        return jsonify({
            'success': True,
            'prediction': result,
            'image': img_data,
            'gradcam': gradcam_b64,
            'lime': lime_b64
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    print("Starting OSCC Detection Flask app...")
    if load_app_model():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("=" * 50)
        print("FATAL: Model failed to load.")
        print("=" * 50)
