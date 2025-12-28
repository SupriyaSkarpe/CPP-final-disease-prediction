from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ===== LOAD MODELS =====
heart_model = joblib.load("models/heart_rf_model.pkl")

diabetes_model = joblib.load("models/diabetes_lr_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

# ===== HOME =====
@app.get("/")
def home():
    return {"message": "Disease Prediction API Running"}

# ===== HEART =====
class HeartInput(BaseModel):
    features: list[float]

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    if len(data.features) != 11:
        raise HTTPException(status_code=400, detail="Heart model expects 11 features")

    X = np.array(data.features).reshape(1, -1)
    prediction = heart_model.predict(X)[0]
    return {"prediction": int(prediction)}

# ===== DIABETES =====
class DiabetesInput(BaseModel):
    features: list[float]

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    if len(data.features) != 17:
        raise HTTPException(status_code=400, detail="Diabetes model expects 17 features")

    X = np.array(data.features).reshape(1, -1)
    X_scaled = diabetes_scaler.transform(X)
    prediction = diabetes_model.predict(X_scaled)[0]
    return {"prediction": int(prediction)}
