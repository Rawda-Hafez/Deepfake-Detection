import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# =========================
# UI Styling
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1 {
    text-align: center;
    color: #22d3ee;
    font-weight: 800;
}

.block-container {
    padding-top: 2rem;
}

.stFileUploader {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Model Path
# =========================
MODEL_PATH = "../models/deepfake_resnet_model.keras"

# =========================
# Load Model Safely
# =========================
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("✅ Model Loaded Successfully!")
else:
    st.error("❌ Model file not found! Check path.")
    st.stop()

IMG_SIZE = (224, 224)

# =========================
# Title
# =========================
st.title("🧠 Deepfake Detection App")
st.write("Upload an image and AI will detect if it's REAL or FAKE 🚀")

# =========================
# Prediction Function
# =========================
def predict_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x, verbose=0)[0][0]

    # 🔥 FIX: invert logic
    if pred > 0.5:
        return "REAL ✅", float(pred)
    else:
        return "FAKE ❌", float(1 - pred)
# =========================
# Upload Section
# =========================
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # ================= Image =================
    with col1:
        st.markdown("### 📷 Input Image")
        st.image(image, use_container_width=True)

    # ================= Prediction =================
    label, confidence = predict_image(img_array)

    with col2:
        st.markdown("### 🔍 Prediction Result")

        if label == "REAL ✅":
            st.success(label)
        else:
            st.error(label)

        st.markdown("### 🎯 Confidence")
        st.progress(confidence)

        st.write(f"**Score:** {confidence * 100:.2f}%")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("🚀 Built with Deep Learning | ResNet50 | Streamlit")
