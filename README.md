# 🎭 Deepfake Detection System

A deep learning project for detecting AI-generated (Deepfake) faces versus authentic images using **ResNet50 Transfer Learning**, featuring both a modern **Streamlit GUI application** and a **Flask Web API** with multiple frontend design themes.

---

## 🌟 Features

- **Deep Learning Model**: ResNet50-based binary classification model fine-tuned for facial authenticity detection.
- **Interactive Streamlit App**:
  - Animated, responsive UI with glassmorphism design.
  - Drag-and-drop image upload.
  - Real-time prediction with confidence scores.
- **Flask REST API & Web Interfaces**:
  - Standalone Flask backend (`/predict` endpoint).
  - 5 distinctive frontend design templates (Ethereal, Elite, Standard, Neural, Noir).
- **Comprehensive Jupyter Notebooks**: Full exploratory data analysis, CNN baseline training, ResNet50 transfer learning, and evaluation metrics (Confusion Matrix, ROC/AUC).

---

## 📁 Project Structure

```text
Deepfake-Detection/
├── app/
│   └── app.py                     # Streamlit GUI Application
├── Web/
│   ├── server.py                  # Flask API server
│   ├── Design (1) .html           # Ethereal Theme
│   ├── Design (2) .html           # Elite Theme
│   ├── Design (3) .html           # Standard Theme
│   ├── Design (4) .html           # Neural Edition Theme
│   └── Design (5) .html           # Noir Theme
├── models/
│   └── best_deepfake_detector.keras # Trained model weights (place here)
├── deepfake.ipynb                 # Full EDA, CNN baseline & ResNet50 Transfer Learning notebook
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation
└── LICENSE                        # Project license
```

---

## 📊 Dataset

The dataset contains real and synthetically generated faces.
You can download the dataset from Google Drive:
👉 [Download Dataset](https://drive.google.com/drive/folders/10J_SwQs9oAmcqp80aJsbUH11ugPXnFva)

Extract it into the `dataset/` directory before training or testing locally.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Setup
Ensure the trained model `best_deepfake_detector.keras` is placed inside the `models/` folder:
```text
models/best_deepfake_detector.keras
```

---

## 💻 How to Run

### Option A: Streamlit GUI Application
Run the Streamlit app:
```bash
streamlit run app/app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### Option B: Flask API + Web Frontend
1. Start the Flask backend server:
   ```bash
   python Web/server.py
   ```
   The API will listen at `http://localhost:5000/predict`.

2. Open any of the HTML designs in `Web/` (e.g., `Design (1) .html` to `Design (5) .html`) in your favorite browser.

---



---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
