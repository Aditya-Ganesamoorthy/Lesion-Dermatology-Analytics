import streamlit as st

st.set_page_config(
    page_title="Skin Cancer AI Predictor",
    layout="centered"
)

import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np

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

# STREAMLIT APP
st.title("🩺 Skin Cancer Single Image Prediction")
st.caption("Academic demo — not for clinical diagnosis")

uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]

    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    predicted_dx = CLASS_NAMES[pred_idx]
    risk = get_risk(predicted_dx)
    recommendation = get_recommendation(risk)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Diagnosis:** `{predicted_dx}`")
    st.write(f"**Confidence:** `{confidence:.2%}`")
    st.write(f"**Risk Category:** `{risk}`")
    st.write(f"**Clinical Recommendation:** {recommendation}")

    st.subheader("Class-wise Probabilities")
    sorted_indices = np.argsort(preds)[::-1]

    for i in sorted_indices:
        st.write(f"{CLASS_NAMES[i]}: {preds[i]:.2%}")

    st.warning(
        "⚠ This system is for academic demonstration only and does not replace professional medical diagnosis."
    )
