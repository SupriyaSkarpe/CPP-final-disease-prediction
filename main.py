from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ---------------- LOAD MODELS ----------------
heart_model = joblib.load("models/heart_rf_model.pkl")

diabetes_model = joblib.load("models/diabetes_lr_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

# ---------------- SCHEMAS ----------------
class HeartInput(BaseModel):
    features: list

class DiabetesInput(BaseModel):
    features: list

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "Disease Prediction API Running"}

# -------- HEART --------
@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = int(heart_model.predict(X)[0])
    return {"prediction": prediction}

# -------- DIABETES --------
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = diabetes_scaler.transform(X)
    prediction = int(diabetes_model.predict(X_scaled)[0])
    return {"prediction": prediction}
