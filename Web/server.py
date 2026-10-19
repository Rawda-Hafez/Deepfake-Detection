from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import os
import io
from PIL import Image

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ==========================================
# Load Detection Engine
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_deepfake_detector.keras")
IMG_SIZE = (224, 224)

model = None
if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("🚀 Neural Forensics Engine Online: Model Loaded Successfully")
    except Exception as e:
        try:
            import keras
            model = keras.models.load_model(MODEL_PATH, compile=False)
            print("🚀 Neural Forensics Engine Online (via Keras direct): Model Loaded Successfully")
        except Exception as e2:
            print(f"🚨 ERROR loading model: {e2}")
else:
    print(f"🚨 ERROR: Model file not found at {MODEL_PATH}!")

# ==========================================
# Web App Entry Route
# ==========================================
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# ==========================================
# Prediction API Route
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Detection model not loaded on server"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No image payload found"}), 400
    
    file = request.files['file']
    img_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(image)

        # Preprocessing: convert RGB -> BGR for opencv resize then normalize
        img_processed = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_processed = cv2.resize(img_processed, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

        tensor = img_rgb.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        # Prediction
        prediction = float(model.predict(tensor, verbose=0)[0][0])

        if prediction > 0.5:
            label = "AUTHENTIC"
            confidence = float(prediction * 100)
        else:
            label = "SYNTHETIC"
            confidence = float((1 - prediction) * 100)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 1),
            "raw_score": round(prediction, 4),
            "status": "success"
        })
    except Exception as err:
        return jsonify({"error": str(err)}), 500

# ==========================================
# Run Server
# ==========================================
if __name__ == '__main__':
    print("Serving Veritas AI on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)