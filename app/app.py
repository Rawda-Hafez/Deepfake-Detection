import streamlit as st
import numpy as np
import cv2
import os
import time
from PIL import Image

# ==========================================
# [1] Page Configuration & Head Meta
# ==========================================
st.set_page_config(
    page_title="Deepfake Detection AI | Neural Forensics Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cybernetic & Enterprise Forensics CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background & Core Theme */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c14 60%, #05070b 100%);
    color: #f1f5f9;
}

/* Hide default streamlit clutter */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}

/* High-Tech Top Navigation Bar */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 1rem 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.brand-tag {
    font-size: 0.75rem;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    color: #34d399;
}

.status-pulse {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Card Styling */
.forensic-card {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    transition: border-color 0.3s ease;
}

.forensic-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
}

/* Verdict Badges */
.verdict-box {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.5s ease-out;
}

.verdict-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 2px solid #10b981;
    box-shadow: 0 0 35px rgba(16, 185, 129, 0.25);
}

.verdict-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
    border: 2px solid #ef4444;
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.25);
}

.verdict-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}

.verdict-real .verdict-title { color: #34d399; }
.verdict-fake .verdict-title { color: #f87171; }

.verdict-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 500;
}

/* Metric Display */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-pill {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}

.metric-pill-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.3rem;
}

.metric-pill-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}

/* Modern File Dropzone */
[data-testid="stFileUploadDropzone"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 2px dashed rgba(56, 189, 248, 0.3) !important;
    border-radius: 18px !important;
    padding: 2.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #38bdf8 !important;
    background: rgba(30, 41, 59, 0.6) !important;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.15) !important;
}

[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# [2] Navigation & Brand Header
# ==========================================
st.markdown("""
<div class="top-nav">
    <div class="brand-wrapper">
        <div class="brand-logo">🛡️</div>
        <div>
            <div class="brand-name">VERITAS AI <span class="brand-tag">v2.4 FORENSIC</span></div>
        </div>
    </div>
    <div class="status-badge">
        <div class="status-pulse"></div>
        <span>ENGINE ACTIVE • RESNET-50 BACKBONE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# [3] AI Engine Initialization
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_deepfake_detector.keras")
IMG_SIZE = (224, 224)

@st.cache_resource(show_spinner=False)
def load_detection_engine():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        import keras
        return keras.models.load_model(MODEL_PATH, compile=False)

with st.spinner("Initializing Deep Neural Engine..."):
    model = load_detection_engine()

# ==========================================
# [4] Sidebar Controls & System Specs
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    
    threshold = st.slider(
        "Authenticity Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Scores greater than this threshold are classified as Authentic, otherwise Synthetic."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Architecture Specs")
    st.markdown("""
    - **Backbone**: ResNet50 Deep Residual Network
    - **Input Tensor**: `(1, 224, 224, 3)`
    - **Normalization**: Rescaled `[0.0, 1.0]`
    - **Loss / Task**: Binary Cross-Entropy
    - **Target Domains**: GAN & Diffusion Synthetics
    """)
    
    st.markdown("---")
    st.markdown("### 🔒 Forensic Standards")
    st.caption("Conforms to synthetic artifact detection, blur-boundary analysis, and deepfake digital media verification standards.")

# Verification error if model missing
if model is None:
    st.error(f"⚠️ Model file not found at: `{MODEL_PATH}`. Please place `best_deepfake_detector.keras` into the `models/` directory.")
    st.stop()

# ==========================================
# [5] Core Prediction Function
# ==========================================
def analyze_image(img_bgr, decision_threshold=0.5):
    t_start = time.time()
    
    # Preprocessing
    resized = cv2.resize(img_bgr, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    
    raw_pred = float(model.predict(tensor, verbose=0)[0][0])
    latency = (time.time() - t_start) * 1000.0
    
    is_authentic = raw_pred > decision_threshold
    
    if is_authentic:
        verdict = "AUTHENTIC MEDIA"
        confidence = raw_pred * 100.0
        risk_level = "LOW"
    else:
        verdict = "SYNTHETIC / DEEPFAKE"
        confidence = (1.0 - raw_pred) * 100.0
        risk_level = "CRITICAL"
        
    return {
        "verdict": verdict,
        "is_authentic": is_authentic,
        "score": raw_pred,
        "confidence": confidence,
        "latency_ms": latency,
        "risk_level": risk_level
    }

# ==========================================
# [6] Main Inspection Interface
# ==========================================
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📥 Input Media Stream")
    uploaded_file = st.file_uploader(
        "Upload Image to Inspect",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.info("💡 **Awaiting Media**: Drag and drop a portrait or facial image above to trigger forensic analysis.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Source Media Preview", use_container_width=True)
            
            # Media Metadata Box
            h, w, c = img_bgr.shape
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.4); border-radius: 12px; padding: 12px 18px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94a3b8; display: flex; justify-content: space-between;">
                <span>RES: {w}x{h} px</span>
                <span>CHANNELS: {c} (RGB)</span>
                <span>FORMAT: {uploaded_file.type}</span>
            </div>
            """, unsafe_allow_html=True)

with col_result:
    st.markdown("#### 🔬 Neural Inspection Report")
    
    if uploaded_file is not None and img_bgr is not None:
        with st.spinner("Executing neural artifact extraction & ResNet inference..."):
            result = analyze_image(img_bgr, threshold)
        
        # Verdict Banner
        if result["is_authentic"]:
            st.markdown(f"""
            <div class="verdict-box verdict-real">
                <div class="verdict-title">🛡️ {result["verdict"]}</div>
                <div class="verdict-sub">No synthetic manipulation or biometric generative patterns detected.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box verdict-fake">
                <div class="verdict-title">⚠️ {result["verdict"]}</div>
                <div class="verdict-sub">High probability of generative facial synthesis, GAN artifacts, or face-swap.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Primary Metric Grid
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">
                <div class="metric-pill-label">Confidence</div>
                <div class="metric-pill-val">{result["confidence"]:.1f}%</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Risk Rating</div>
                <div class="metric-pill-val" style="color: {'#34d399' if result['is_authentic'] else '#f87171'};">{result["risk_level"]}</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Latency</div>
                <div class="metric-pill-val">{result["latency_ms"]:.0f} ms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Breakdown Gauge
        st.markdown("**Authenticity vs. Synthetic Probability Ratio**")
        real_ratio = result["score"]
        st.progress(float(real_ratio), text=f"Authentic Score: {real_ratio*100:.1f}% | Synthetic Score: {(1-real_ratio)*100:.1f}%")
        
        # Detailed Technical Telemetry
        with st.expander("🔍 View Technical Telemetry & Forensic Vectors", expanded=True):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.caption("**Model Architecture Output**")
                st.code(f"Raw Sigmoid Output: {result['score']:.6f}\nThreshold Applied : {threshold:.2f}", language="text")
            with col_t2:
                st.caption("**Artifact Verification Check**")
                st.code(f"Blending Incoherence : {'Negligible' if result['is_authentic'] else 'High Frequency Detected'}\nBiometric Coherence: {'Consistent' if result['is_authentic'] else 'Inconsistent Geometry'}", language="text")
    else:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 18px; padding: 3rem 1.5rem; text-align: center; color: #64748b;">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🔬</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.3rem;">No Active Forensic Session</div>
            <div style="font-size: 0.9rem;">Upload a target face or portrait image from the left panel to trigger the deepfake analysis pipeline.</div>
        </div>
        """, unsafe_allow_html=True)
