<div align="center">

# 👤 VisAge AI - Deep Facial Age & Gender Intelligence

[![Live Demo](https://img.shields.io/badge/Live%20Demo-https%3A%2F%2Fvisage--ai.onrender.com-007acc?style=for-the-badge&logo=render&logoColor=white)](https://visage-ai.onrender.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005A9E?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![VS Code Theme](https://img.shields.io/badge/UI_Theme-VS_Code_Blue_%26_White-007acc?style=for-the-badge&logo=visualstudiocode&logoColor=white)](#-user-interface--features)

*An end-to-end Computer Vision & Deep Multi-Task Convolutional Neural Network (MobileNetV2 Transfer Learning) platform for real-time age estimation and gender binary classification.*

</div>

---

## 🚀 Live Demo
Access the live web application deployed on Render:
👉 **[https://visage-ai.onrender.com](https://visage-ai.onrender.com)**

---

## 🌟 Key Features

- **Multi-Task Neural Network:** Joint estimation of continuous Age regression (L1 MAE Loss) and binary Gender classification (Binary Cross-Entropy Loss) using a shared pre-trained MobileNetV2 backbone.
- **High Performance:** Achieves **0.9597 ROC-AUC**, **87.77% Test Accuracy** (92.58% Validation / 95.16% Train), and **6.43 years Validation Age MAE** (7.74y baseline).
- **Computer Vision Pipeline:** Pre-processed face detection and landmark alignment via OpenCV Haar Cascade (`haarcascade_frontalface_default`).
- **VS Code Signature UI Theme:** Glassmorphic single-page web app styled with VS Code Blue & White design palette, featuring live browser webcam feeds (`navigator.mediaDevices`), drag-and-drop file uploads, test set sample galleries, and a Light/Dark theme toggle.
- **RESTful API Backend:** Fast async FastAPI web server providing `/api/health`, `/api/predict`, `/api/predict-base64`, and `/api/sample-images` endpoints.
- **ONNX Runtime Export:** Optimized cross-platform model inference (`saved_models/best_model.onnx`) verified against PyTorch outputs.
- **Containerized & Cloud Ready:** Ready-to-deploy `Dockerfile`, `docker-compose.yml`, `Procfile`, and 1-click VS Code `launch.json` debugging setup.

---

## 📊 Quantitative Model Benchmarks

### Test Set Evaluation (2,371 Unseen Test Faces)
Evaluated on a 10% held-out test split from 23,708 aligned facial images in the UTKFace dataset:

| Category | Metric | Score / Result |
| :--- | :--- | :---: |
| **Gender Classification** | **Accuracy** | **87.77%** |
| | **ROC AUC** | **0.9597** |
| | **Precision** | **89.49%** |
| | **Recall** | **84.28%** |
| | **F1 Score** | **0.8681** |
| **Age Regression** | **Mean Absolute Error (MAE)** | **10.53 years** (Val MAE: **6.43 yrs**) |
| | **Root Mean Squared Error (RMSE)** | **14.92 years** |
| | **R² Determination Score** | **0.4362** |

### Fine-Tuning Epoch Progression
| Stage | Validation Loss | Validation Gender Acc | Train Gender Acc | Validation Age MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Head Warmup** | `0.6578` | `73.01%` | `70.59%` | `11.76 yrs` |
| **Phase 1 Complete (Epoch 8)** | `0.5422` | `78.62%` | `77.85%` | `10.01 yrs` |
| **Phase 2 Fine-Tune (Epoch 12)** | `0.3929` | `86.42%` | `86.45%` | `8.75 yrs` |
| **Phase 2 Fine-Tune (Epoch 20)** | `0.3348` | `88.36%` | `89.10%` | `7.74 yrs` |
| **Extended Fine-Tune (Epoch 36)** | `0.2549` | `92.49%` | `94.28%` | `6.50 yrs` |
| **Extended Fine-Tune (Best)** | **`0.2525`** | **`92.58%`** | **`95.16%`** | **`6.43 yrs`** |

---

## 📁 Project Structure

```text
VisAge-AI/
├── app/
│   ├── main.py              # FastAPI Web Application & REST API Endpoints
│   └── static/
│       ├── index.html       # Glassmorphic VS Code Blue & White Single-Page App
│       ├── style.css        # VS Code Light & Dark Theme Design System
│       └── app.js           # Frontend Webcam, Drag-and-Drop, API Logic
├── src/
│   ├── __init__.py
│   ├── dataset.py           # UTKFace PyTorch Loader & Augmentation Pipeline
│   ├── model.py             # Multi-Task MobileNetV2 Architecture Design
│   ├── train.py             # Multi-Epoch Chunked Training & Fine-Tuning Engine
│   ├── evaluate.py          # Metric Generator (Accuracy, Precision, MAE, R²)
│   └── predict.py           # OpenCV Face Detection & Inference Engine
├── saved_models/
│   ├── best_model.pt        # Trained PyTorch Model State Dict
│   └── best_model.onnx      # Exported & Verified ONNX Model Weights
├── .vscode/
│   └── launch.json          # 1-Click VS Code F5 Run & Debug Configurations
├── export_onnx.py           # ONNX Exporter & Runtime Verification Script
├── streamlit_app.py         # Streamlit Dashboard Interface
├── Dockerfile               # Container Image Configuration
├── docker-compose.yml       # Docker Orchestration Configuration
├── Procfile                 # Cloud Web Service Start Command
├── requirements.txt         # Dependencies List
└── README.md                # Documentation & Architecture Overview
```

---

## 💻 Running in VS Code

### 1. Open Folder in VS Code
Open VS Code, press `Ctrl + K, Ctrl + O`, and select the project directory.

### 2. Run via Terminal (`Ctrl + ~`)
- **FastAPI Web Dashboard (Recommended):**
  ```bash
  python -m uvicorn app.main:app --reload --port 8000
  ```
  *Open your browser to: `http://localhost:8000`*

- **Streamlit Dashboard:**
  ```bash
  streamlit run streamlit_app.py
  ```

- **OpenCV Webcam Live Window:**
  ```bash
  python -m src.predict
  ```

### 3. 1-Click Debugger (`F5`)
Press `Ctrl + Shift + D`, select **`🚀 Run FastAPI Web Server (App)`**, and press **`F5`**!

---

## ⚡ REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | System status, model load state, and active hardware device |
| `/api/predict` | `POST` | Upload multipart form-data image file for face detection & prediction |
| `/api/predict-base64` | `POST` | Send base64-encoded image string (webcam feed or sample) |
| `/api/sample-images` | `GET` | Fetch test set sample images for quick UI testing |

### Example API Request (Python)
```python
import requests

url = "https://visage-ai.onrender.com/api/predict"
files = {"file": open("my_face.jpg", "rb")}
response = requests.post(url, files=files).json()

print(f"Gender: {response['gender']} ({response['gender_confidence']}%)")
print(f"Age: {response['age']} years ({response['age_group']})")
```

---

## 🐳 Docker Deployment

To build and run the Docker container locally or on a remote server:

```bash
# Build and launch background container
docker-compose up --build -d

# Stop container
docker-compose down
```

---

## 📜 License & Acknowledgments

- **Dataset:** Trained on the [UTKFace Dataset](https://susanqq.github.io/UTKFace/) (23,708 facial images).
- **License:** MIT License. Built with PyTorch, MobileNetV2, OpenCV, and FastAPI.
