from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from diet import router as diet_router

app = FastAPI(title="Disease Prediction API with Human Explainable AI")
app.include_router(diet_router)


heart_rf = joblib.load("models/heart_rf_model.pkl")
heart_knn = joblib.load("models/heart_knn.pkl")
heart_knn_scaler = joblib.load("models/heart_knn_scaler.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

diabetes_lr = joblib.load("models/diabetes_lr_model.pkl")
diabetes_knn = joblib.load("models/diabetes_knn_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
DIABETES_FEATURES = joblib.load("models/diabetes_features.pkl")
heart_svm = joblib.load("models/heart_svm.pkl")
heart_svm_scaler = joblib.load("models/heart_svm_scaler.pkl")

heart_xgb = joblib.load("models/heart_xgb.pkl")
heart_catboost = joblib.load("models/heart_catboost.pkl")
heart_dt = joblib.load("models/heart_dt.pkl")
heart_gb = joblib.load("models/heart_gb.pkl")

heart_rf_explainer = shap.TreeExplainer(heart_rf)


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



def human_explanation(feature, impact):
    direction = "increases" if impact > 0 else "reduces"

    feature_map = {
        "Cholesterol": "cholesterol level",
        "RestingBP": "blood pressure",
        "MaxHR": "heart rate",
        "Age": "age",
        "Oldpeak": "heart stress during exercise",
        "FastingBS_1": "blood sugar level",
        "ExerciseAngina_Y": "chest pain during exercise",
        "ST_Slope_Flat": "ECG heart pattern",
        "Sex_M": "male gender"
    }

    readable = feature_map.get(feature, feature.replace("_", " "))
    return f"Your {readable} {direction} the risk of heart disease."



@app.post("/predict/heart/all")
def predict_all_heart(data: HeartInput):
    try:
        X = pd.DataFrame([data.dict()])
        X = pd.get_dummies(X)
        X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

        results = {}

        # RF
        rf_pred = int(heart_rf.predict(X)[0])
        rf_prob = float(heart_rf.predict_proba(X)[0][1])
        results["Random Forest"] = {
            "prediction": rf_pred,
            "probability": round(rf_prob * 100, 2)
        }

        # KNN
        X_knn = heart_knn_scaler.transform(X)
        knn_pred = int(heart_knn.predict(X_knn)[0])
        knn_prob = float(heart_knn.predict_proba(X_knn)[0][1])
        results["KNN"] = {
            "prediction": knn_pred,
            "probability": round(knn_prob * 100, 2)
        }

        # SVM
        X_svm = heart_svm_scaler.transform(X)
        svm_pred = int(heart_svm.predict(X_svm)[0])
        svm_prob = float(heart_svm.predict_proba(X_svm)[0][1])
        results["SVM"] = {
            "prediction": svm_pred,
            "probability": round(svm_prob * 100, 2)
        }

        # XGB
        xgb_pred = int(heart_xgb.predict(X)[0])
        xgb_prob = float(heart_xgb.predict_proba(X)[0][1])
        results["XGBoost"] = {
            "prediction": xgb_pred,
            "probability": round(xgb_prob * 100, 2)
        }

        # CatBoost
        cat_pred = int(heart_catboost.predict(X)[0])
        cat_prob = float(heart_catboost.predict_proba(X)[0][1])
        results["CatBoost"] = {
            "prediction": cat_pred,
            "probability": round(cat_prob * 100, 2)
        }

        # DT
        dt_pred = int(heart_dt.predict(X)[0])
        dt_prob = float(heart_dt.predict_proba(X)[0][1])
        results["Decision Tree"] = {
            "prediction": dt_pred,
            "probability": round(dt_prob * 100, 2)
        }

        # GB
        gb_pred = int(heart_gb.predict(X)[0])
        gb_prob = float(heart_gb.predict_proba(X)[0][1])
        results["Gradient Boosting"] = {
            "prediction": gb_pred,
            "probability": round(gb_prob * 100, 2)
        }

        best_model = max(results.items(), key=lambda x: x[1]["probability"])

        return {
            "disease": "Heart Disease",
            "results": results,
            "best_model": best_model[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))