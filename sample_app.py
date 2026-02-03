import streamlit as st

# ============================================================
# STREAMLIT CONFIG (MUST BE FIRST)
# ============================================================
st.set_page_config(
    page_title="Skin Cancer AI Predictor",
    layout="centered"
)

# ============================================================
# IMPORTS
# ============================================================
import tensorflow as tf
import numpy as np
from PIL import Image
import subprocess

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = (224, 224)
MODEL_PATH = "skin_cancer_multiclass_model.h5"

CONFIDENCE_THRESHOLD = 0.50      # relaxed
ENTROPY_THRESHOLD = 1.40         # uncertainty cutoff

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

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
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

def compute_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

# ============================================================
# OLLAMA AI EXPLANATION
# ============================================================
def generate_ai_explanation(dx, confidence, risk):
    prompt = f"""
        A dermoscopic image was analyzed using a deep learning model.

        Predicted lesion type: {dx}
        Model confidence: {confidence:.2f}
        Risk category: {risk}

        Generate a concise clinical explanation intended for healthcare professionals.
        Do NOT provide diagnosis.
        Do NOT recommend treatment, biopsy, or medication.
        Do NOT address the patient directly.
        Focus only on risk interpretation and general clinical consideration.
        """

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout.strip()

# ============================================================
# UI
# ============================================================
st.title("🩺 Skin Cancer AI – Single Image Prediction")
st.caption("Academic healthcare analytics decision-support system")

uploaded_file = st.file_uploader(
    "Upload a dermoscopic skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- CNN Prediction ----------------
    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    entropy = compute_entropy(preds)

    # ---------------- OOD DECISION ----------------
    is_ood = (
        confidence < CONFIDENCE_THRESHOLD and
        entropy > ENTROPY_THRESHOLD
    )

    if is_ood:
        st.error("⚠ Unsupported or non-dermoscopic image detected.")
        st.warning(
            "The uploaded image does not appear to be a valid dermoscopic skin lesion. "
            "Please upload a proper skin lesion image."
        )
        st.info(
            f"Model uncertainty too high (Entropy: {entropy:.2f}, "
            f"Confidence: {confidence:.2%})."
        )
        st.stop()

    # ---------------- VALID PREDICTION ----------------
    predicted_dx = CLASS_NAMES[pred_idx]
    risk = get_risk(predicted_dx)
    recommendation = get_recommendation(risk)

    st.subheader("🔍 Prediction Summary")
    st.write(f"**Predicted Lesion Type:** `{predicted_dx}`")
    st.write(f"**Confidence Score:** `{confidence:.2%}`")
    st.write(f"**Risk Category:** `{risk}`")
    st.write(f"**Clinical Recommendation:** {recommendation}")

    # ---------------- AI EXPLANATION ----------------
    with st.spinner("🧠 Generating AI clinical explanation..."):
        ai_explanation = generate_ai_explanation(
            predicted_dx,
            confidence,
            risk
        )

    st.subheader("🧠 AI Clinical Explanation")
    st.write(ai_explanation)

    # ---------------- CLASS PROBABILITIES ----------------
    st.subheader("📊 Class-wise Probabilities")
    for i in np.argsort(preds)[::-1]:
        st.write(f"{CLASS_NAMES[i]}: {preds[i]:.2%}")

    st.info(
        "⚠ This system is intended strictly for academic decision-support purposes "
        "and does not provide medical diagnosis or treatment."
    )
