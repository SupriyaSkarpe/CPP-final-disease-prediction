from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Load model & columns
dt_model = joblib.load("models/heart_dt_optimized.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

@app.post("/test-dt")
def test_dt(data: dict):
    try:
        # Convert input to DataFrame
        X = pd.DataFrame([data])

        # One-hot encode
        X = pd.get_dummies(X)

        # Align columns with training
        X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

        # Prediction
        pred = dt_model.predict(X)[0]

        return {"prediction": int(pred)}

    except Exception as e:
        return {"error": str(e)}
