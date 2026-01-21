from fastapi import FastAPI
import pandas as pd
import xgboost as xgb
import joblib

app = FastAPI()

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/heart_xgb.json")

HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

@app.post("/test-heart-xgb")
def test_xgb(data: dict):

    X = pd.DataFrame([data])
    X = pd.get_dummies(X)
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

    pred = xgb_model.predict(X)[0]

    return {
        "model": "XGBoost",
        "prediction": int(pred)
    }
