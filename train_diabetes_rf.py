import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# Replace 0 values with median (important step)
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
# TRAIN MODEL
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = rf.predict(X_test)

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
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = rf.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# =========================
# SAVE MODEL
# =========================
joblib.dump(rf, "diabetes_rf_model.pkl")

print("\nModel saved successfully!")