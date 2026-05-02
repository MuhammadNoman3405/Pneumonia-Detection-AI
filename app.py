import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pneumonia AI - UET Taxila",
    page_icon="🏥",
    layout="centered"
)

# --- PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .title-text { text-align: center; color: #1e3d59; font-weight: bold; }
    .author-info { text-align: center; color: #555; font-style: italic; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- RESEARCH TITLE ---
st.markdown("<h1 class='title-text'>Pneumonia Detection from Chest X-Ray Images using Deep Learning with ROI-based Preprocessing and Class Imbalance Handling</h1>", unsafe_allow_html=True)

# --- AUTHOR & INSTITUTION INFO ---
# Information derived from Research Paper [cite: 5, 8, 10, 12, 13]
st.markdown(f"""
    <div class='author-info'>
    <b>Department of Computer Science | University of Engineering and Technology, Taxila</b><br>
    Submitted to: <b>Dr. Munwar Iqbal</b> <br>
    <b>Research Team:</b><br>
    Muhammad Noman (23-CS-68) | Muhammad Junaid (23-CS-66) | Muhammad Haris Tahir (23-CS-106)
    </div>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    model_path = 'best_pneumonia_model.h5'
    return tf.keras.models.load_model(model_path)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- MAIN INTERFACE ---
st.write("---")
st.markdown("### 📥 Patient Data Input")
uploaded_file = st.file_uploader("Upload a Chest X-Ray (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# --- PROCESSING LOGIC ---
# Preprocessing pipeline as described in Methodology [cite: 365, 387, 394]
def preprocess_roi_image(img):
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    
    # ROI Enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization) [cite: 365, 387]
    img_uint8 = img.astype('uint8')
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)
    
    # Converting back to 3 channels for MobileNetV2/EfficientNet [cite: 401]
    final_img = cv2.merge([enhanced, enhanced, enhanced])
    return final_img.astype('float32') / 255.0

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.markdown("#### **Original X-Ray**")
        st.image(image, use_container_width=True)
    
    # Run Diagnostic
    if st.button("🔍 START ROI-BASED DIAGNOSTIC ANALYSIS"):
        with st.spinner('Applying ROI Preprocessing & Analyzing...'):
            processed_img = preprocess_roi_image(image)
            
            with col2:
                st.markdown("#### **ROI Enhanced View**")
                st.image(processed_img, use_container_width=True)
            
            # Prediction Logic [cite: 403, 412]
            processed_img_input = np.expand_dims(processed_img, axis=0)
            prediction = model.predict(processed_img_input)
            confidence = prediction[0][0]

            st.write("---")
            st.markdown("### 🧬 AI Diagnostic Result")
            
            if confidence > 0.5:
                st.error(f"## **Result: PNEUMONIA DETECTED**")
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
            else:
                st.success(f"## **Result: NORMAL (HEALTHY)**")
                st.metric(label="Confidence Score", value=f"{(1-confidence)*100:.2f}%")
            
            # Technical Details from Paper [cite: 50, 51]
            with st.expander("View Technical Evaluation (Metrics)"):
                st.write("This model was evaluated using:")
                st.write("- **ROI-based Preprocessing** (CLAHE + Gaussian Blur) [cite: 365]")
                st.write("- **Class-Weighted Loss** to handle dataset imbalance (2.9:1 ratio) [cite: 345, 375]")
                st.write("- **Metrics:** Accuracy, Recall, Specificity, and AUC-ROC [cite: 51]")

st.write("---")
st.caption("⚠️ Disclaimer: This system is a research prototype developed for CS-307. It is not intended for official medical diagnosis.")
