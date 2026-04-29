import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Page Config
st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# app.py mein line 21 ko is tarah update karain:
@st.cache_resource
def load_model():
    # 'r' lagana zaroori hai Windows path ke liye
    model_path = r'C:\Users\DELL\Desktop\Pneumonia_WebApp\best_pneumonia_model.h5'
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("🩺 AI Chest X-Ray Analyzer")
st.write("UET Taxila - Research Project | Developer: Noman")
st.write("---")

uploaded_file = st.file_uploader("Upload a Patient's Chest X-Ray (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Image Display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)
    
    # 2. Preprocessing (Proposed Method)
    def preprocess_image(img):
        img = np.array(img.convert('RGB'))
        img = cv2.resize(img, (224, 224))
        # ROI Enhancement (CLAHE)
        img_uint8 = img.astype('uint8')
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_gray)
        final_img = cv2.merge([enhanced, enhanced, enhanced])
        return final_img.astype('float32') / 255.0

    if st.button("Run Diagnostic AI"):
        with st.spinner('Analyzing patterns...'):
            processed_img = preprocess_image(image)
            processed_img = np.expand_dims(processed_img, axis=0)
            
            prediction = model.predict(processed_img)
            confidence = prediction[0][0]

            st.write("---")
            if confidence > 0.5:
                st.error(f"### Result: PNEUMONIA DETECTED")
                st.write(f"**Confidence Level:** {confidence*100:.2f}%")
            else:
                st.success(f"### Result: NORMAL")
                st.write(f"**Confidence Level:** {(1-confidence)*100:.2f}%")
            
            st.info("Note: This is an AI-assisted tool. Please consult a radiologist for final diagnosis.")