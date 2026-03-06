import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("dataset/diabetes_pima.csv")

# ===============================
# Replace 0 with NaN (medical logic)
# ===============================
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# ===============================
# Median Imputation
# ===============================
df.fillna(df.median(), inplace=True)

# ===============================
# Split Features & Target
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Logistic Regression (BEST SAFE CONFIG)
# ===============================
lr = LogisticRegression(
    C=1.0,                 # balanced regularization
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs"         # modern & stable
)

lr.fit(X_train_scaled, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = lr.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("🔥 Logistic Regression Accuracy:", round(accuracy * 100, 2), "%")
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Save Model
# ===============================
joblib.dump(lr, "models/diabetes_lr_model_pima.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")

print("\n✅ Diabetes LR model & scaler saved")
