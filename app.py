"""
Pneumonia Detection - Flask API Server
Pipeline: Receive imageURL → Fetch → Preprocess → Predict → Return result
"""

import os
import io
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Allow requests from Express backend

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "./model/pneumonia_cnn.keras")  # or .h5
IMG_SIZE   = (224, 224)   # Must match what you trained with (e.g. 224x224)
CLASSES    = ["NORMAL", "PNEUMONIA"]

# ─────────────────────────────────────────
# LOAD MODEL ONCE ON STARTUP
# ─────────────────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")


# ─────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────
def preprocess_image_from_url(image_url: str) -> np.ndarray:
    """
    Fetch image from URL (sent by Express), resize, normalize,
    and return as a batch-ready numpy array.
    """
    # 1. Fetch image bytes
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()

    # 2. Open with Pillow and convert to RGB (handles grayscale X-rays too)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    # 3. Resize to match model input
    img = img.resize(IMG_SIZE)

    # 4. Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # 5. Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_from_file(file) -> np.ndarray:
    """
    Alternative: Accept raw file upload instead of URL.
    Useful for direct uploads without cloud storage.
    """
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ─────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────
def run_prediction(img_array: np.ndarray) -> dict:
    """
    Run model inference and return prediction + confidence score.
    Handles both binary (sigmoid) and multi-class (softmax) output.
    """
    raw_output = model.predict(img_array)

    # Binary classification (sigmoid output, single neuron)
    if raw_output.shape[-1] == 1:
        confidence = float(raw_output[0][0])
        predicted_class = CLASSES[1] if confidence >= 0.5 else CLASSES[0]
        # Normalize confidence to reflect the predicted class
        confidence_score = confidence if predicted_class == "PNEUMONIA" else 1 - confidence

    # Multi-class (softmax output, 2 neurons)
    else:
        confidence_scores = raw_output[0]
        predicted_index   = int(np.argmax(confidence_scores))
        predicted_class   = CLASSES[predicted_index]
        confidence_score  = float(confidence_scores[predicted_index])

    return {
        "prediction":      predicted_class,          # "PNEUMONIA" or "NORMAL"
        "confidenceScore": round(confidence_score * 100, 2),  # e.g. 94.73 (%)
    }


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Express can ping this to confirm Python server is alive."""
    return jsonify({"status": "ok", "model": MODEL_PATH}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main endpoint. Called by Express after JWT is validated.

    Expected JSON body from Express:
    {
        "imageURL": "https://your-storage/xray123.jpg"
    }

    Returns:
    {
        "prediction": "PNEUMONIA",
        "confidenceScore": 94.73
    }
    """
    data = request.get_json()

    if not data or "imageURL" not in data:
        return jsonify({"error": "Missing 'imageURL' in request body"}), 400

    image_url = data["imageURL"]

    try:
        # Run full pipeline
        img_array = preprocess_image_from_url(image_url)
        result    = run_prediction(img_array)
        return jsonify(result), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 422

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict-upload", methods=["POST"])
def predict_upload():
    """
    Alternative endpoint for direct file uploads (no URL needed).
    Useful during development or if you skip cloud storage.

    Form-data: file = <xray image>
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        img_array = preprocess_image_from_file(file)
        result    = run_prediction(img_array)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)