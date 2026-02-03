import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Skin Cancer AI Predictor",
    layout="centered"
)

import tensorflow as tf
import numpy as np
from PIL import Image
import subprocess
import warnings
warnings.filterwarnings("ignore")

# CONFIG
IMG_SIZE = (224, 224)
MODEL_PATH = "skin_cancer_multiclass_model.h5"

CLASS_NAMES = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

HIGH_RISK = {"mel", "bcc", "akiec"}
MODERATE_RISK = {"bkl"}
LOW_RISK = {"nv", "df", "vasc"}

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# HELPER FUNCTIONS
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def get_risk(dx):
    if dx in HIGH_RISK:
        return "High Risk"
    elif dx in MODERATE_RISK:
        return "Moderate Risk"
    else:
        return "Low Risk"

def get_recommendation(risk):
    if risk == "High Risk":
        return "Immediate dermatology consultation recommended"
    elif risk == "Moderate Risk":
        return "Clinical follow-up advised"
    else:
        return "Routine monitoring suggested"

# OLLAMA AI EXPLANATION FUNCTION
def generate_ai_explanation(dx, confidence, risk):
    prompt = f"""
A dermoscopic image was analyzed using a deep learning model.

Predicted lesion type: {dx}
Model confidence: {confidence:.2f}
Risk category: {risk}

Generate a concise clinical explanation for a doctor.
Do NOT confirm diagnosis.
Do NOT prescribe medication.
Focus on risk interpretation and recommended next steps.
"""
    
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout.strip()

# STREAMLIT UI
st.title("🩺 Skin Cancer AI – Single Image Prediction")
st.caption("Academic demo for healthcare analytics (not for clinical diagnosis)")

uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # CNN Prediction
    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])

    predicted_dx = CLASS_NAMES[pred_idx]
    risk = get_risk(predicted_dx)
    recommendation = get_recommendation(risk)

    # DISPLAY CORE RESULTS
    st.subheader("Prediction Result")
    st.write(f"**Predicted Diagnosis:** `{predicted_dx}`")
    st.write(f"**Confidence:** `{confidence:.2%}`")
    st.write(f"**Risk Category:** `{risk}`")
    st.write(f"**Clinical Recommendation:** {recommendation}")

    

    # CLASS PROBABILITIES
    st.subheader("📊 Class-wise Probabilities")
    sorted_indices = np.argsort(preds)[::-1]

    for i in sorted_indices:
        st.write(f"{CLASS_NAMES[i]}: {preds[i]:.2%}")

    # DISCLAIMER
    st.info(
        "⚠ This system is intended strictly for academic decision-support demonstration. "
        "It does not provide medical diagnosis or treatment."
    )
