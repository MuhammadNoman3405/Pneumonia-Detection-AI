# 🩺 AI Chest X-Ray Analyzer — Pneumonia Detection

A deep learning-powered web application for automated **Pneumonia Detection from Chest X-Ray Images**, built as a research project for the **CS-307 Artificial Neural Networks** course at **UET Taxila**. Upload a chest X-ray and get an instant AI-assisted diagnosis with confidence score.

---

## 🔬 Research Overview

This project implements the methodology from our research paper:

> **"Pneumonia Detection from Chest X-Ray Images using Deep Learning with ROI-based Preprocessing and Class Imbalance Handling"**
> — Department of Computer Science, UET Taxila | Spring 2026

The model uses **MobileNetV2 transfer learning** trained on the Kaggle Chest X-Ray Pneumonia dataset (5,863 images), with **class-weighted binary cross-entropy** to handle the 2.9:1 class imbalance. ROI-based preprocessing using **CLAHE contrast enhancement** is applied to isolate lung regions before classification.

---

## 🚀 Features

- **🔍 AI Diagnosis:** Upload any chest X-ray (JPG/PNG) and get an instant Normal or Pneumonia prediction
- **📊 Confidence Score:** Displays model confidence percentage for each prediction
- **🫁 ROI Preprocessing:** CLAHE-based contrast enhancement applied before inference for better lung structure visibility
- **⚡ Fast Inference:** Results in seconds using cached MobileNetV2 model
- **⚠️ Clinical Disclaimer:** Built-in reminder to consult a radiologist for final diagnosis

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Base Model** | MobileNetV2 (ImageNet pretrained, frozen) |
| **Total Parameters** | 2,422,081 |
| **Trainable Parameters** | 163,968 (custom head only) |
| **Input Size** | 224 × 224 × 3 |
| **Classification Head** | GAP → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid) |
| **Loss Function** | Binary Cross-Entropy with Class Weights |
| **Optimizer** | Adam (lr = 0.0001) |
| **Training Epochs** | 30 (Early Stopping, patience=5) |

---

## 📈 Performance on Test Set (624 images)

| Metric | Normal | Pneumonia |
|---|---|---|
| **Precision** | 0.88 | 0.89 |
| **Recall** | 0.80 | 0.94 |
| **F1-Score** | 0.84 | 0.91 |
| **Overall Accuracy** | — | **88%** |
| **Weighted F1** | — | **0.88** |

**Confusion Matrix:**
- ✅ True Positives (Pneumonia): 365
- ✅ True Negatives (Normal): 187
- ❌ False Negatives (missed pneumonia): 25
- ⚠️ False Positives: 47

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Deep Learning:** TensorFlow 2.15.0, Keras
- **Image Processing:** OpenCV (CLAHE, grayscale, resize), Pillow
- **Numerical:** NumPy < 2.0.0
- **Runtime:** Python 3.11
- **Deployment:** Hugging Face Spaces

---

## 📁 Project Structure

```
pneumonia-detection/
├── app.py                      # Streamlit web application
├── best_pneumonia_model.h5     # Trained MobileNetV2 model weights
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version specification
└── README.md
```

---

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone https://huggingface.co/spaces/<your-username>/pneumonia-detection
cd pneumonia-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
tensorflow==2.15.0
opencv-python-headless
pillow
numpy<2.0.0
```

---

## ⚠️ Disclaimer

This application is an **AI-assisted research tool** developed for academic purposes. It should **not** be used as a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for final diagnosis and treatment decisions.

---

## 👥 Development Team

| Name | Reg No | Role |
|---|---|---|
| **Muhammad Noman** | 23-CS-68 | App Development & Model Integration |
| Muhammad Junaid | 23-CS-66 | Model Training & Evaluation |
| Muhammad Haris Tahir | 23-CS-106 | Data Preprocessing & Research |

**Supervised by:** Dr. Muhammad Munwar Iqbal
**Department:** Computer Science, UET Taxila
**Course:** CS-307 — Artificial Neural Networks | Spring 2026

---

## 📚 Dataset

- **Source:** [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney (2018)
- **Original Paper:** Kermany et al. (2018) — *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 172(5), 1122–1131
- **Total Images:** 5,863 (Train: 5,216 | Val: 16 | Test: 624)
- **Classes:** Normal (1,341 train) vs Pneumonia (3,875 train)

---

*Developed with ❤️ by Team 23-CS | BSCS @ UET Taxila | Spring 2026*
