import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("dataset/diabetes1.csv")

print("Dataset shape:", df.shape)

# ===============================
# 2. Handle Missing / Invalid Values
# ===============================
# Replace 0 with median ONLY for clinical columns
zero_cols = ['BMI', 'SBP', 'DBP', 'FPG', 'Chol', 'Tri', 'HDL', 'LDL', 'ALT', 'BUN', 'CCR', 'FFPG']

for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

# ===============================
# 3. Split Features & Target
# ===============================
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 5. Feature Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 6. Logistic Regression Model
# ===============================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train_scaled, y_train)

# ===============================
# 7. Evaluation
# ===============================
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Diabetes Logistic Regression Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_score = cross_val_score(
    model,
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross-Validation Accuracy:", cv_score.mean())

# ===============================
# 8. Save Model & Scaler
# ===============================
joblib.dump(model, "models/diabetes_lr_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")

print("Diabetes model & scaler saved successfully!")
