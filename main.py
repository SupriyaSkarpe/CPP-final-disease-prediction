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
#heart_svm = joblib.load("models/heart_svm_best.pkl")
#heart_svm_scaler = joblib.load("models/heart_svm_scaler.pkl")

#heart_lr = joblib.load("models/heart_lr.pkl")

heart_dt = joblib.load("models/heart_decision_tree_best.pkl")
#heart_gb = joblib.load("models/heart_gb.pkl")
#heart_xgb = joblib.load("models/heart_xgb.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")



heart_rf_explainer = shap.TreeExplainer(heart_rf)

def preprocess_heart_input(data):
    X = pd.DataFrame([data.dict()])

    # Apply one-hot encoding
    X = pd.get_dummies(X)

    # IMPORTANT: match training columns exactly
    X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

    return X


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



@app.post("/predict/heart/{model_name}")
def predict_heart(model_name: str, data: HeartInput):

    try:
        X = preprocess_heart_input(data)

        if model_name == "rf":
            prediction = heart_rf.predict(X)[0]

        elif model_name == "knn":
            X_scaled = heart_knn_scaler.transform(X.values)  # ✅ FIX
            prediction = heart_knn.predict(X_scaled)[0]

        else:
            raise HTTPException(status_code=400, detail="Use rf or knn")

        return {
            "disease": "Heart Disease",
            "model_used": model_name,
            "prediction": int(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-all/heart")
def predict_all_heart(data: HeartInput):
    try:
        X = preprocess_heart_input(data)

        results = {}

        # Random Forest
        rf_prob = heart_rf.predict_proba(X)[0][1]
        results["Random Forest"] = round(rf_prob * 100, 2)

        # KNN
        X_scaled = heart_knn_scaler.transform(X.values)
        knn_prob = heart_knn.predict_proba(X_scaled)[0][1]
        results["KNN"] = round(knn_prob * 100, 2)

        # Logistic Regression
        #lr_prob = heart_lr.predict_proba(X)[0][1]
        #esults["Logistic Regression"] = round(lr_prob * 100, 2)

        # SVM
        #svm_prob = heart_svm.predict_proba(X)[0][1]
        #results["SVM"] = round(svm_prob * 100, 2)

        # Decision Tree
        dt_prob = heart_dt.predict_proba(X)[0][1]
        results["Decision Tree"] = round(dt_prob * 100, 2)

        # Gradient Boosting
        #gb_prob = heart_gb.predict_proba(X)[0][1]
        #results["Gradient Boosting"] = round(gb_prob * 100, 2)

        # XGBoost
        #xgb_prob = heart_xgb.predict_proba(X)[0][1]
        #results["XGBoost"] = round(xgb_prob * 100, 2)

        return {
            "disease": "Heart Disease",
            "all_model_predictions (%)": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-explain/heart/rf")
def predict_explain_heart_rf(data: HeartInput):

    try:
        X = pd.DataFrame([data.dict()])
        X = pd.get_dummies(X)
        X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

        prediction = int(heart_rf.predict(X)[0])
        probability = float(heart_rf.predict_proba(X)[0][1])

        label = "Yes" if prediction == 1 else "No"

        # ✅ SAFE SHAP (no crash)
        try:
            shap_values = heart_rf_explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]

            explanations = []

            for feature, impact in zip(HEART_COLUMNS, shap_vals):
                explanations.append({
                    "reason": human_explanation(feature, impact),
                    "impact": round(float(impact), 4)
                })

            explanations = sorted(
                explanations,
                key=lambda x: abs(x["impact"]),
                reverse=True
            )[:5]

        except Exception as e:
            explanations = [{"reason": "Explanation unavailable", "impact": 0}]

        return {
            "disease": "Heart Disease",
            "prediction": label,
            "probability": round(probability * 100, 2),
            "xai": {
                "method": "SHAP",
                "reasons": explanations
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-full/heart")
def predict_full_heart(data: HeartInput):
    try:
        X = preprocess_heart_input(data)

        results = {}

        # ===== Predictions =====
        rf_prob = heart_rf.predict_proba(X)[0][1]
        results["Random Forest"] = round(rf_prob * 100, 2)

        X_scaled = heart_knn_scaler.transform(X.values)
        knn_prob = heart_knn.predict_proba(X_scaled)[0][1]
        results["KNN"] = round(knn_prob * 100, 2)

        dt_prob = heart_dt.predict_proba(X)[0][1]
        results["Decision Tree"] = round(dt_prob * 100, 2)

        # ===== SHAP Explanation (ONLY RF) =====
        try:
            shap_values = heart_rf_explainer(X)
            shap_vals = shap_values.values[0]

            explanations = []

            for feature, impact in zip(HEART_COLUMNS, shap_vals):
                explanations.append({
                    "reason": human_explanation(feature, impact),
                    "impact": round(float(impact), 4)
                })

            explanations = sorted(
                explanations,
                key=lambda x: abs(x["impact"]),
                reverse=True
            )[:5]

        except:
            explanations = [{"reason": "Explanation unavailable", "impact": 0}]

        return {
            "disease": "Heart Disease",
            "predictions": results,
            "explanation_model": "Random Forest",
            "explanations": explanations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))