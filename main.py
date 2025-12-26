from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("models/heart_rf_model.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API running"}

class HeartInput(BaseModel):
    features: list

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]

    return {
        "prediction": int(prediction)
    }
