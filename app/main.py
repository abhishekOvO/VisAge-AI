import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.predict import AgeGenderPredictor

app = FastAPI(
    title="Age & Gender Prediction API",
    description="High-Accuracy Facial Analysis using Deep Convolutional Neural Networks (MobileNetV2)",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor Engine
MODEL_PATH = os.path.join("saved_models", "best_model.pt")
predictor = AgeGenderPredictor(model_path=MODEL_PATH)

# Mount Static Files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class Base64ImageRequest(BaseModel):
    image_base64: str

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Age & Gender Prediction API is running.</h1>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": os.path.exists(MODEL_PATH),
        "device": str(predictor.device),
        "architecture": "MobileNetV2 Multi-Task CNN"
    }

def process_prediction(cv_img):
    result = predictor.predict(cv_img)
    
    # Annotate bounding box on image for visualization
    if result["face_detected"] and result["bbox"]:
        bx, by, bw, bh = result["bbox"]
        color = (52, 211, 153) if result["gender"] == "Female" else (59, 130, 246)
        cv2.rectangle(cv_img, (bx, by), (bx + bw, by + bh), color, 3)
        
        label = f"{result['gender']}, {result['age']}y ({result['gender_confidence']}%)"
        cv2.putText(cv_img, label, (bx, max(25, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Convert annotated image to base64 for UI display
    rgb_annotated = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_annotated = Image.fromarray(rgb_annotated)
    buf = io.BytesIO()
    pil_annotated.save(buf, format="JPEG", quality=90)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    
    result["annotated_image"] = annotated_b64
    return result

@app.post("/api/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result = process_prediction(cv_img)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.post("/api/predict-base64")
async def predict_base64(payload: Base64ImageRequest):
    try:
        if not payload.image_base64:
            raise HTTPException(status_code=400, detail="Empty image_base64 string.")
        b64_data = payload.image_base64.split(",")[-1]
        img_bytes = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result = process_prediction(cv_img)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 processing error: {str(e)}")

@app.get("/api/sample-images")
async def get_sample_images():
    sample_dir = r"C:\Users\ashid\Documents\all_utkface"
    samples = []
    if os.path.exists(sample_dir):
        files = os.listdir(sample_dir)[:12]
        for f in files:
            p = os.path.join(sample_dir, f)
            if os.path.isfile(p):
                parts = f.split('_')
                if len(parts) >= 2:
                    try:
                        age = parts[0]
                        gender = "Female" if parts[1] == "1" else "Male"
                        img = Image.open(p).convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                        samples.append({
                            "name": f,
                            "true_age": age,
                            "true_gender": gender,
                            "image_b64": b64
                        })
                    except:
                        continue
    return JSONResponse(content={"samples": samples})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
