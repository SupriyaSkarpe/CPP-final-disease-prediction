import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset/diabetes_pima.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# =========================
# DATA CLEANING
# =========================
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING (IMPORTANT)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# TRAIN LOGISTIC REGRESSION
# =========================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# =========================
# PREDICTION
# =========================
y_pred = lr.predict(X_test_scaled)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# SAVE MODEL + SCALER
# =========================
joblib.dump(lr, "models/diabetes_lr_model.pkl")
joblib.dump(scaler, "models/diabetes_lr_scaler.pkl")

print("\nLogistic Regression model saved successfully!")