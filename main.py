from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# =====================================
# CREATE APP
# =====================================
app = FastAPI(title="Disease Prediction API")

# =====================================
# LOAD MODELS (ONCE)
# =====================================

# HEART MODELS
heart_rf = joblib.load("models/heart_rf_model.pkl")
#heart_gb = joblib.load("models/heart_gb.pkl")
heart_knn = joblib.load("models/heart_knn.pkl")
heart_knn_scaler = joblib.load("models/heart_knn_scaler.pkl")

# Columns used during training (VERY IMPORTANT)
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

# DIABETES MODELS
diabetes_model = joblib.load("models/diabetes_lr_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

# =====================================
# HOME ROUTE
# =====================================
@app.get("/")
def home():
    return {
        "message": "Heart & Diabetes Disease Prediction API is running"
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

    # Convert input to DataFrame
    X = pd.DataFrame([data.dict()])

    # One-hot encoding
    X = pd.get_dummies(X)

    # Align columns with training
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

    # Select model
    if model_name == "rf":
        prediction = heart_rf.predict(X)[0]

   # elif model_name == "gb":
    #    prediction = heart_gb.predict(X)[0]

    elif model_name == "knn":
        X_scaled = heart_knn_scaler.transform(X)
        prediction = heart_knn.predict(X_scaled)[0]

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. Use: rf, knn"
        )

    return {
        "disease": "Heart Disease",
        "model_used": model_name,
        "prediction": int(prediction)
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