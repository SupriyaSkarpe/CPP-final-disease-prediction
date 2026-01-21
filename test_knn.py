from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load KNN model + scaler
heart_knn = joblib.load("models/heart_knn.pkl")
heart_knn_scaler = joblib.load("models/heart_knn_scaler.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

@app.post("/test-heart-knn")
def test_heart_knn(data: dict):
    X = pd.DataFrame([data])
    X = pd.get_dummies(X)
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)
    X_scaled = heart_knn_scaler.transform(X)
    pred = heart_knn.predict(X_scaled)[0]
    return {"prediction": int(pred)}
