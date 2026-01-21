from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Heart LR Test API")

# ==========================
# LOAD LR MODEL + METADATA
# ==========================
heart_lr = joblib.load("models/heart_lr_model.pkl")
heart_lr_scaler = joblib.load("models/heart_lr_scaler.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

# ==========================
# TEST ENDPOINT
# ==========================
@app.post("/test-heart-lr")
def test_heart_lr(data: dict):

    # Convert input to DataFrame
    X = pd.DataFrame([data])

    # One-hot encode
    X = pd.get_dummies(X)

    # Align columns
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

    # Scale (LR NEEDS scaling)
    X_scaled = heart_lr_scaler.transform(X)

    # Predict
    prediction = heart_lr.predict(X_scaled)[0]

    return {
        "model": "Logistic Regression",
        "prediction": int(prediction)
    }
