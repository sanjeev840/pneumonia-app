"""Flask inference API for Pneumonia Detection.
GET  /health   -> status + model input shape
POST /predict  -> multipart form field "file" (png/jpg/jpeg/dcm)
"""
import io
import os

import cv2
import numpy as np
import pydicom
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "model/pneumonia_model.keras")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
CLASS_NAMES = ["Normal", "Pneumonia"]          # sigmoid output = P(Pneumonia)
ALLOWED = {".png", ".jpg", ".jpeg", ".dcm"}

app = Flask(__name__)
CORS(app)

# Load the model once at startup
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
_, H, W, C = model.input_shape                 # e.g. (None, 128, 128, 1)
print(f"Model loaded. Input shape: {model.input_shape}", flush=True)


def read_pixels(file_bytes: bytes, filename: str) -> np.ndarray:
    """Return a 2-D float32 grayscale array from png/jpg/dcm bytes."""
    if filename.lower().endswith(".dcm"):
        img = pydicom.dcmread(io.BytesIO(file_bytes)).pixel_array.astype(np.float32)
    else:
        img = np.asarray(Image.open(io.BytesIO(file_bytes)).convert("L"), dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """Replicates the notebook's load_and_preprocess()."""
    img = cv2.resize(img, (W, H))
    img = img / max(float(img.max()), 1e-8)    # per-image normalisation to [0,1]
    x = img[..., np.newaxis]                   # (H, W, 1)
    if C == 3:                                 # VGG16 / ResNet50 models
        x = np.repeat(x, 3, axis=-1)
        x = tf.keras.applications.vgg16.preprocess_input(x * 255.0)
    return np.expand_dims(x, 0).astype(np.float32)


@app.get("/health")
def health():
    return jsonify(status="ok", model_input_shape=list(model.input_shape))


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify(error="No file uploaded (form field must be 'file')."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type '{ext}'. Use {sorted(ALLOWED)}"), 400
    try:
        x = preprocess(read_pixels(f.read(), f.filename))
        p_pneu = float(np.asarray(model.predict(x, verbose=0)).reshape(-1)[0])
    except Exception as e:
        return jsonify(error=f"Could not process image: {e}"), 422

    label = CLASS_NAMES[int(p_pneu >= THRESHOLD)]
    return jsonify(
        predicted_class=label,
        probability_pneumonia=round(p_pneu, 4),
        probability_normal=round(1 - p_pneu, 4),
        threshold=THRESHOLD,
    )


if __name__ == "__main__":                     # local debugging only
    app.run(host="0.0.0.0", port=5000)
