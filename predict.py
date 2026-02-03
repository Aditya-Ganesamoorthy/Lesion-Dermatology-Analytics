import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model #type: ignore
# IMPORT generator AND validation dataframe
from images_preprocessing import val_generator, val_df
import warnings
warnings.filterwarnings("ignore")


MODEL_PATH = "skin_cancer_multiclass_model.h5"
model = load_model(MODEL_PATH)

print("\n Model loaded successfully")

# CLASS LABEL Mapping
CLASS_LABELS = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# RUN PREDICTIONS
print("\n Running predictions on validation set...")

pred_probs = model.predict(val_generator, verbose=1)

# EXTRACT RESULTS
predicted_indices = np.argmax(pred_probs, axis=1)
predicted_dx = [CLASS_LABELS[i] for i in predicted_indices]
confidence_scores = np.max(pred_probs, axis=1)

# BUILD RESULTS DATAFRAME (SAFE WAY)
results_df = val_df.copy().reset_index(drop=True)

results_df["predicted_dx"] = predicted_dx
results_df["confidence"] = confidence_scores

results_df.to_csv("prediction_results.csv", index=False)

print("Predictions saved to prediction_results.csv")

################################

df_res = pd.read_csv("prediction_results.csv")

# CLINICAL REFERRAL LOGIC
def referral_decision(dx):
    if dx in ["mel", "bcc"]:
        return "Immediate dermatologist referral"
    elif dx in ["akiec"]:
        return "Specialist evaluation recommended"
    elif dx in ["bkl"]:
        return "Clinical review advised"
    else:
        return "Routine monitoring"

# APPLY LOGIC
df_res["referral_recommendation"] = df_res["predicted_dx"].apply(referral_decision)

# OPTIONAL: RISK CATEGORY
def risk_category(dx):
    if dx in ["mel", "bcc"]:
        return "High Risk"
    elif dx in ["akiec", "bkl"]:
        return "Moderate Risk"
    else:
        return "Low Risk"

df_res["risk_category"] = df_res["predicted_dx"].apply(risk_category)

df_res.to_csv("final_prescriptive_data.csv", index=False)

print("\nClinical referral logic applied")
print("Final dataset saved as final_prescriptive_data.csv")