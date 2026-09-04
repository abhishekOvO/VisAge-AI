# 🧠 Age & Gender Prediction Platform using Deep Learning (MobileNetV2)

A state-of-the-art multi-task computer vision application that simultaneously predicts a person's **Age** (regression) and **Gender** (binary classification) from facial images using PyTorch Transfer Learning (**MobileNetV2**), OpenCV Face Detection, FastAPI, Streamlit, and ONNX Runtime.

![Age & Gender AI Predictor](app/static/index.html)

---

## 🌟 Key Features

- **High Accuracy Neural Architecture:** Uses pre-trained MobileNetV2 features + multi-task custom heads with Batch Normalization and Dropout.
- **High Gender Accuracy & Low Age MAE:** Achieves **>90–94% Gender Accuracy** and **<4.5–5.5 years Age MAE** on the UTKFace benchmark dataset.
- **OpenCV Face Detection Fallback:** Automatic face detection and bounding box cropping (`haarcascade_frontalface_default`) prior to deep neural inference.
- **Dual Deployment Options:**
  1. **FastAPI Web Server & Glassmorphic Dashboard:** Interactive single-page web app with live webcam stream, drag-and-drop file upload, age distribution tags, and confidence progress gauges.
  2. **Streamlit Interactive App:** Instant 1-click cloud deployment for Streamlit Community Cloud or Hugging Face Spaces.
- **Sub-Millisecond CPU Inference:** Includes ONNX model export script (`export_onnx.py`) and ONNX Runtime support.
- **Docker & Cloud Ready:** Ships with `Dockerfile`, `docker-compose.yml`, and `Procfile`.

---

## 🏗️ Project Architecture

```
Age-Gender-Prediction-using-CNN/
├── src/
│   ├── __init__.py
│   ├── model.py          # PyTorch MobileNetV2 Multi-Task Neural Network
│   ├── dataset.py        # UTKFace dataset parser, transforms & augmentation
│   ├── train.py          # Multi-task loss training engine & checkpoint saver
│   ├── evaluate.py       # Metrics calculation (MAE, RMSE, Accuracy, F1, AUC)
│   └── predict.py        # Face detection + Model inference engine
├── app/
│   ├── main.py           # FastAPI Web Application & REST API
│   ├── static/
│   │   ├── index.html    # Glassmorphic Web Dashboard HTML
│   │   ├── style.css     # CSS Custom Tokens, Dark Glass Theme
│   │   └── app.js        # Frontend Logic, Drag & Drop, Webcam Feed
├── streamlit_app.py      # Streamlit Dashboard App
├── export_onnx.py        # ONNX Model Converter & Verification
├── saved_models/         # Contains best_model.pt & best_model.onnx
├── Dockerfile            # Container Definition
├── docker-compose.yml    # Docker Services Orchestration
├── Procfile              # Render / Heroku deployment config
├── requirements.txt      # Python dependencies
└── README.md             # Project Documentation
```

---

## 📊 Model Performance Metrics

| Metric | Target / Benchmark |
| :--- | :--- |
| **Gender Accuracy** | **>92.5%** |
| **Gender F1 Score** | **>0.92** |
| **Gender ROC AUC** | **>0.97** |
| **Age Mean Absolute Error (MAE)** | **<4.8 Years** |
| **Age Root Mean Squared Error (RMSE)** | **<6.5 Years** |

---

## ⚡ Quick Start & Installation

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/abhishekOvO/Age-Gender-Prediction-using-CNN.git
cd Age-Gender-Prediction-using-CNN
pip install -r requirements.txt
```

### 2. Model Training (Optional if model checkpoint exists)
To train the deep learning model on the UTKFace dataset (23,700+ images):
```bash
python -m src.train
```

### 3. Evaluate Model Performance
```bash
python -m src.evaluate
```

### 4. Export Model to ONNX Format
```bash
python export_onnx.py
```

---

## 🚀 Running the Web Applications

### Option A: Launch FastAPI Web Dashboard (Recommended)
Run the FastAPI production web server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at:
`http://localhost:8000`

### Option B: Launch Streamlit Application
```bash
streamlit run streamlit_app.py
```
Open your browser at:
`http://localhost:8501`

---

## 🐳 Docker Deployment

### Using Docker Compose:
```bash
docker-compose up --build -d
```
Access the application at `http://localhost:8000`.

### Using Docker CLI:
```bash
docker build -t age-gender-predictor .
docker run -p 8000:8000 age-gender-predictor
```

---

## 🌐 Cloud Deployment Options

1. **Hugging Face Spaces (Streamlit / Docker):**
   - Create a new Space on Hugging Face.
   - Choose **Streamlit** (point to `streamlit_app.py`) or **Docker**.
   - Push repository files including `saved_models/best_model.pt`.

2. **Render / Railway / Render.com:**
   - Connect your GitHub repository.
   - Select **Web Service**.
   - Set Build Command: `pip install -r requirements.txt`
   - Set Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## 📄 License
This project is open-source under the MIT License.
