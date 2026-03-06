import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("dataset/heart.csv")
print("Dataset shape:", df.shape)

# ===============================
# 2. Data Cleaning
# ===============================
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ===============================
# 3. Encoding
# ===============================
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

# ===============================
# 4. Split Features & Target
# ===============================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 5. Scaling (important)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 6. Define Base Models
# ===============================
lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="liblinear"
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ===============================
# 7. Stacking Ensemble
# ===============================
stack_model = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),
    cv=5,
    n_jobs=-1
)

# ===============================
# 8. Train
# ===============================
stack_model.fit(X_train, y_train)

# ===============================
# 9. Evaluation
# ===============================
y_pred = stack_model.predict(X_test)
y_prob = stack_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🔥 Stacking Ensemble Accuracy:", round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 10. Save Model
# ===============================
joblib.dump(stack_model, "models/heart_stacking_model.pkl")
joblib.dump(scaler, "models/heart_scaler.pkl")

print("✅ Stacking model & scaler saved successfully!")
