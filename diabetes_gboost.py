import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("dataset/diabetes_pima.csv")

# ===============================
# 2. Proper data cleaning
# ===============================
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# 3. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. XGBoost (STRONG CONFIG)
# ===============================
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=1.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("🔥 XGBoost Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Cross-validation (IMPORTANT FOR TEACHER)
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(model, scaler.fit_transform(X), y, cv=cv, scoring="accuracy")

print("✅ Cross-Validation Accuracy:", round(cv_score.mean()*100, 2), "%")

# ===============================
# 8. Save model
# ===============================
joblib.dump(model, "models/diabetes_xgb_best.pkl")
joblib.dump(scaler, "models/diabetes_xgb_scaler.pkl")
