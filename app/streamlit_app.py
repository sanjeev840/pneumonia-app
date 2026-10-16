
import io
import sys
import os
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Dynamically insert possible app folder paths into sys.path so 'preprocessing' can be found
possible_paths = [
    os.path.join(os.getcwd(), "app"),
    "/content/drive/MyDrive/GL/Capstone/deployment_files/backend_files",
    "/content/drive/MyDrive/GL/Capstone"
]
for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from preprocessing import preprocess_for_inference
except ModuleNotFoundError:
    # Fallback skeleton function if preprocessing.py is not reachable
    st.warning("preprocessing.py module not found in path. Using fallback preprocessing function.")
    def preprocess_for_inference(raw_array):
        import cv2
        img = cv2.resize(raw_array, (128, 128))
        img = img.astype(np.float32) / (img.max() if img.max() > 0 else 1.0)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        import tensorflow as tf
        img = tf.keras.applications.vgg16.preprocess_input(img * 255.0)
        return np.expand_dims(img, axis=0)

st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("Chest X-Ray Pneumonia Detection")
st.caption("Capstone demo — not for clinical use.")

@st.cache_resource          # model loads once per container, not per request
def get_model():
    return load_model("/content/drive/MyDrive/GL/Capstone/best_model.keras")

try:
    model = get_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    model = None

THRESHOLD = 0.5   # change if your ROC analysis suggested a different operating point

uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["dcm", "png", "jpg", "jpeg"],
)

if uploaded is not None:
    # 1. Read — branch on file type
    if uploaded.name.lower().endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(uploaded.read()))
        raw = ds.pixel_array
    else:
        raw = np.array(Image.open(uploaded).convert("L"))   # grayscale

    # 2. Preprocess — EXACTLY as training
    batch = preprocess_for_inference(raw)                   # shape: (1, H, W, C)

    # 3. Predict
    if model is not None:
        prob = float(model.predict(batch, verbose=0).ravel()[0])
        pred_label = "Pneumonia" if prob >= THRESHOLD else "Normal"

        # 4. Display
        col1, col2 = st.columns(2)
        with col1:
            st.image(raw, caption="Uploaded X-ray", use_container_width=True, clamp=True)
        with col2:
            st.metric("Prediction", pred_label)
            st.metric("P(pneumonia)", f"{prob:.3f}")
            st.progress(prob)
    else:
        st.error("Model is not loaded. Cannot run prediction.")
