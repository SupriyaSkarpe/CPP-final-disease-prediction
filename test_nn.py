from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

heart_nn = joblib.load("models/heart_nn.pkl")
heart_nn_scaler = joblib.load("models/heart_nn_scaler.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

@app.post("/test-heart-nn")
def test_nn(data: dict):

    X = pd.DataFrame([data])
    X = pd.get_dummies(X)
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)
    X_scaled = heart_nn_scaler.transform(X)

    pred = heart_nn.predict(X_scaled)[0]

    return {
        "model": "Neural Network",
        "prediction": int(pred)
    }
