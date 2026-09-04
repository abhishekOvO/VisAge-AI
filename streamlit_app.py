import os
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from src.predict import AgeGenderPredictor

# Page Configuration
st.set_page_config(
    page_title="Age & Gender AI Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stAppHeader { background-color: transparent; }
    .css-1r650q0 { background-color: #121a2c; }
    .metric-card {
        background-color: #121a2c;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    model_path = os.path.join("saved_models", "best_model.pt")
    return AgeGenderPredictor(model_path=model_path)

predictor = load_predictor()

# Header
st.title("🧠 Age & Gender AI Prediction Platform")
st.caption("Powered by PyTorch, MobileNetV2 Transfer Learning & OpenCV Face Detection")

# Sidebar Controls
st.sidebar.header("⚙️ Settings & Input Source")
input_option = st.sidebar.radio("Select Input Source:", ["Upload Image", "Take Camera Photo", "Pick Sample Image"])

input_image = None

if input_option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose a facial image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

elif input_option == "Take Camera Photo":
    camera_photo = st.sidebar.camera_input("Take a photo")
    if camera_photo is not None:
        input_image = Image.open(camera_photo).convert("RGB")

elif input_option == "Pick Sample Image":
    sample_dir = r"C:\Users\ashid\Documents\all_utkface"
    if os.path.exists(sample_dir):
        sample_files = os.listdir(sample_dir)[:15]
        selected_sample = st.sidebar.selectbox("Choose sample image:", sample_files)
        if selected_sample:
            sample_path = os.path.join(sample_dir, selected_sample)
            input_image = Image.open(sample_path).convert("RGB")

# Main Content Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Image Preview")
    if input_image is not None:
        st.image(input_image, use_container_width=True)
    else:
        st.info("👈 Please select or upload an image from the sidebar to begin analysis.")

with col2:
    st.subheader("📊 AI Analysis & Predictions")
    if input_image is not None:
        with st.spinner("Analyzing facial features and estimating age & gender..."):
            cv_img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
            res = predictor.predict(cv_img)
            
            # Draw bounding box
            if res["face_detected"] and res["bbox"]:
                bx, by, bw, bh = res["bbox"]
                color = (52, 211, 153) if res["gender"] == "Female" else (59, 130, 246)
                cv2.rectangle(cv_img, (bx, by), (bx + bw, by + bh), color, 3)
                
            rgb_annotated = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Display Metrics
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                gender_icon = "♀️" if res["gender"] == "Female" else "♂️"
                st.metric(label=f"{gender_icon} Predicted Gender", value=res["gender"], delta=f"{res['gender_confidence']}% Conf")
                st.progress(res["gender_confidence"] / 100.0)
                
            with m_col2:
                st.metric(label="🎂 Predicted Age", value=f"{res['age']} yrs", delta=f"Range: {res['age_range']}")
                st.write(f"**Life Stage:** `{res['age_group']}`")
                
            st.divider()
            st.image(rgb_annotated, caption="Annotated Face Detection", use_container_width=True)
            
            if res["face_detected"]:
                st.success("✅ Face successfully detected and isolated for inference.")
            else:
                st.warning("⚠️ No distinct face detected by OpenCV. Analyzed full frame.")

st.markdown("---")
st.markdown("Developed with PyTorch MobileNetV2 • Multi-Task Loss Optimization • UTKFace Dataset")
