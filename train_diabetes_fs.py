# ================================
# 10 Years Diabetes Dataset - Logistic Regression
# Full End-to-End Code (Without Feature Selection)
# ================================

# ----------------
# Step 1: Import Libraries
# ----------------
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ----------------
# Step 2: Load Dataset
# ----------------
df = pd.read_csv("C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/diabetes.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# ----------------
# Step 3: Initial Cleaning
# ----------------
# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Drop columns with too many missing values
drop_cols = ['weight', 'payer_code', 'medical_specialty']
df.drop(columns=drop_cols, inplace=True)

# Drop irrelevant ID columns
df.drop(columns=['encounter_id', 'patient_nbr'], inplace=True)

# ----------------
# Step 4: Target Encoding
# ----------------
# Convert readmission to binary: <30 -> 1, >30/NO -> 0
df['readmitted'] = df['readmitted'].map({
    '<30': 1,
    '>30': 0,
    'NO': 0
})

# ----------------
# Step 5: Handle Missing Values
# ----------------
# Fill remaining NaN with mode (most frequent value)
df.fillna(df.mode().iloc[0], inplace=True)

# ----------------
# Step 6: Encode Categorical Features
# ----------------
# Convert categorical variables into numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)
print("Shape after encoding:", df.shape)

# ----------------
# Step 7: Split Features & Target
# ----------------
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# ----------------
# Step 8: Train-Test Split
# ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------
# Step 9: Feature Scaling
# ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------
# Step 10: Train Logistic Regression
# ----------------
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    solver='liblinear'
)
lr_model.fit(X_train_scaled, y_train)

# ----------------
# Step 11: Model Evaluation
# ----------------
y_pred = lr_model.predict(X_test_scaled)
y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------
# Step 12: Save Model & Scaler
# ----------------
joblib.dump(lr_model, "logistic_regression_diabetes.pkl")
joblib.dump(scaler, "scaler_diabetes.pkl")
print("\nModel and scaler saved successfully!")
