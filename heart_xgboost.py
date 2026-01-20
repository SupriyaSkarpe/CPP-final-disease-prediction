import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("dataset/heart.csv")
print("Dataset shape:", df.shape)

# ===============================
# 2. Encode categorical features
# ===============================
df = pd.get_dummies(df, drop_first=True)

# ===============================
# 3. Split X and y
# ===============================
target = "HeartDisease" if "HeartDisease" in df.columns else "target"

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. Feature Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. XGBoost Model
# ===============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n🔥 XGBoost Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC:", round(roc_auc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save Model
# ===============================
joblib.dump(model, "models/heart_xgb_model.pkl")
joblib.dump(scaler, "models/heart_xgb_scaler.pkl")

print("✅ XGBoost model & scaler saved successfully!")
