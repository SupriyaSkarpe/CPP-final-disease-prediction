from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

heart_rf = joblib.load("models/heart_rf_model.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

@app.post("/test-heart")
def test_heart(data: dict):
    X = pd.DataFrame([data])
    X = pd.get_dummies(X)
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)
    pred = heart_rf.predict(X)[0]
    return {"prediction": int(pred)}
