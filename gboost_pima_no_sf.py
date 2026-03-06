import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("dataset/diabetes_pima.csv")
print("Dataset shape:", df.shape)

# ===============================
# 2. Data Cleaning (CRITICAL)
# ===============================
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

# ===============================
# 3. Split X & y
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. Scaling (IMPORTANT for XGBoost stability)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. XGBoost + Hyperparameter Tuning
# ===============================
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum()
)

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

# ===============================
# 6. Evaluation
# ===============================
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🔥 XGBoost Accuracy (No FS):", round(acc*100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save model
# ===============================
joblib.dump(best_xgb, "models/xgb_pima_no_fs.pkl")
joblib.dump(scaler, "models/xgb_pima_scaler.pkl")

print("✅ XGBoost model & scaler saved")
