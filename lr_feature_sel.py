import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# ===============================
# 2. Basic cleaning (minimal)
# ===============================
df = df.drop_duplicates()
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ===============================
# 3. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 4. Scaling (MANDATORY)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. L1 Logistic Regression (Feature Selection)
# ===============================
lasso_lr = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.1,
    max_iter=5000,
    class_weight="balanced"
)

lasso_lr.fit(X_train_scaled, y_train)

# ===============================
# 6. Select important features
# ===============================
coef = lasso_lr.coef_[0]
selected_features = X.columns[coef != 0]

print("Selected Features (L1):")
print(list(selected_features))

# ===============================
# 7. Train final LR on selected features
# ===============================
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

scaler2 = StandardScaler()
X_train_sel = scaler2.fit_transform(X_train_sel)
X_test_sel = scaler2.transform(X_test_sel)

final_lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

final_lr.fit(X_train_sel, y_train)

# ===============================
# 8. Evaluation
# ===============================
y_pred = final_lr.predict(X_test_sel)
acc = accuracy_score(y_test, y_pred)

print("\n🔥 Logistic Regression Accuracy (With Feature Selection):",
      round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9. Save model
# ===============================
joblib.dump(final_lr, "models/heart_lr_l1.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")
joblib.dump(scaler2, "models/heart_scaler_l1.pkl")

print("✅ LR model with L1 feature selection saved")
