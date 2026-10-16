
import io
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from preprocessing import preprocess_for_inference

st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("Chest X-Ray Pneumonia Detection")
st.caption("Capstone demo — not for clinical use.")

@st.cache_resource          # model loads once per container, not per request
def get_model():
    return load_model("app/model/best_model.keras")

model = get_model()

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

    # 2. Preprocess — EXACTLY as training (see step 3)
    batch = preprocess_for_inference(raw)                   # shape: (1, H, W, C)

    # 3. Predict
    prob = float(model.predict(batch, verbose=0).ravel()[0])
    pred_label = "Pneumonia" if prob >= THRESHOLD else "Normal"

    # 4. Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw, caption="Uploaded X-ray", use_column_width=True, clamp=True)
    with col2:
        st.metric("Prediction", pred_label)
        st.metric("P(pneumonia)", f"{prob:.3f}")
        st.progress(prob)
