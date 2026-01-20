import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Load PIMA dataset
# ==============================
df = pd.read_csv("dataset/diabetes1.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# ==============================
# 2. Handle zero values (PIMA rule)
# ==============================
zero_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# ==============================
# 3. Split features & target
# ==============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 4. Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 5. Logistic Regression
# ==============================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train_scaled, y_train)

# ==============================
# 6. Evaluation
# ==============================
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\n🔥 Diabetes Logistic Regression Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 7. Cross-validation
# ==============================
cv_score = cross_val_score(
    model,
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross-Validation Accuracy:", round(cv_score.mean() * 100, 2), "%")

# ==============================
# 8. Save model
# ==============================
joblib.dump(model, "models/diabetes_lr_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")

print("\n✅ Diabetes model & scaler saved successfully!")
