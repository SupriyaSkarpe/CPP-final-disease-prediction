from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

# =====================================
# CREATE APP
# =====================================
app = FastAPI(title="Disease Prediction API with XAI")

# =====================================
# LOAD MODELS (ONCE)
# =====================================

# HEART MODELS
heart_rf = joblib.load("models/heart_rf_model.pkl")
heart_knn = joblib.load("models/heart_knn.pkl")
heart_knn_scaler = joblib.load("models/heart_knn_scaler.pkl")

# Columns used during training
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

# DIABETES MODELS
diabetes_model = joblib.load("models/diabetes_lr_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

# =====================================
# XAI EXPLAINER (CREATE ONCE)
# =====================================
heart_rf_explainer = shap.TreeExplainer(heart_rf)

# =====================================
# HOME ROUTE
# =====================================
@app.get("/")
def home():
    return {
        "message": "Heart & Diabetes Disease Prediction API with Explainable AI is running"
    }

# =====================================
# INPUT SCHEMAS
# =====================================
class HeartInput(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex: str
    ChestPainType: str
    RestingECG: str
    ExerciseAngina: str
    ST_Slope: str


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# =====================================
# HEART PREDICTION (MULTI MODEL)
# =====================================
@app.post("/predict/heart/{model_name}")
def predict_heart(model_name: str, data: HeartInput):

    X = pd.DataFrame([data.dict()])
    X = pd.get_dummies(X)
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

    if model_name == "rf":
        prediction = heart_rf.predict(X)[0]

    elif model_name == "knn":
        X_scaled = heart_knn_scaler.transform(X)
        prediction = heart_knn.predict(X_scaled)[0]

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. Use: rf or knn"
        )

    return {
        "disease": "Heart Disease",
        "model_used": model_name,
        "prediction": int(prediction)
    }

# =====================================
# HEART EXPLAINABLE AI (SHAP)
# =====================================
@app.post("/predict-explain/heart/rf")
def predict_explain_heart_rf(data: HeartInput):

    # ---------------------------
    # Prepare input
    # ---------------------------
    X = pd.DataFrame([data.dict()])
    X = pd.get_dummies(X)
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = int(heart_rf.predict(X)[0])
    probability = float(heart_rf.predict_proba(X)[0][1])

    label = "Yes" if prediction == 1 else "No"

    # ---------------------------
    # SHAP Explanation
    # ---------------------------
    shap_values = heart_rf_explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    shap_vals = shap_vals.reshape(-1)

    explanation = []
    for feature, value in zip(HEART_COLUMNS, shap_vals):
        explanation.append({
            "feature": feature,
            "impact": round(float(value), 4)
        })

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    # ---------------------------
    # Final Response
    # ---------------------------
    return {
        "disease": "Heart Disease",
        "prediction": label,
        "prediction_code": prediction,
        "probability": round(probability * 100, 2),
        "xai": {
            "method": "SHAP (Local Explanation)",
            "top_features": explanation[:5]
        }
    }


# =====================================
# DIABETES PREDICTION
# =====================================
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):

    X = pd.DataFrame([data.dict()])
    X_scaled = diabetes_scaler.transform(X)

    prediction = diabetes_model.predict(X_scaled)[0]

    return {
        "disease": "Diabetes",
        "model_used": "Logistic Regression",
        "prediction": int(prediction)
    }
