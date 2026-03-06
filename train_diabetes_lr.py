# =========================
# Logistic Regression for PIMA Diabetes Dataset
# =========================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("dataset/diabetes_pima.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# =========================
# 2. Replace medically invalid zeros with NaN
# =========================
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# =========================
# 3. Split features and target
# =========================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# 4. Train-test split (stratified)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 5. Handle missing values (Median Imputation)
# =========================
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# =========================
# 6. Feature Scaling (VERY IMPORTANT for LR)
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. Logistic Regression Model
# =========================
model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 8. Prediction
# =========================
y_pred = model.predict(X_test)

# =========================
# 9. Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)
print("\n🔥 Logistic Regression Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 10. Cross-Validation Accuracy
# =========================
X_full = imputer.fit_transform(X)
X_full = scaler.fit_transform(X_full)

cv_scores = cross_val_score(
    model,
    X_full,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation Accuracy:", cv_scores.mean())

# =========================
# 11. Save Model, Scaler & Imputer
# =========================
joblib.dump(model, "models/pima_lr_model.pkl")
joblib.dump(scaler, "models/pima_lr_scaler.pkl")
joblib.dump(imputer, "models/pima_lr_imputer.pkl")

print("\n✅ Logistic Regression model saved successfully!")
