import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# ===============================
# 2. Data cleaning
# ===============================
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ===============================
# 3. One-Hot Encoding
# ===============================
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

# ===============================
# 4. Split X and y
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
# 5. Scaling (MANDATORY)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 6. Feature Selection using L1
# ===============================
lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.1,
    max_iter=5000,
    class_weight="balanced"
)

lasso.fit(X_train_scaled, y_train)

coef = np.abs(lasso.coef_[0])
selected_features = X.columns[coef > 0]

print("🔍 Selected Features:")
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
y_prob = final_lr.predict_proba(X_test_sel)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🔥 Logistic Regression Accuracy (With Feature Selection):",
      round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9. Save model & scalers
# ===============================
joblib.dump(
    final_lr,
    "C:/Users/HP/Documents/CPP-final-disease-prediction/models/heart_lr_fs_model.pkl"
)

joblib.dump(
    scaler2,
    "C:/Users/HP/Documents/CPP-final-disease-prediction/models/heart_lr_fs_scaler.pkl"
)

joblib.dump(
    selected_features.tolist(),
    "C:/Users/HP/Documents/CPP-final-disease-prediction/models/lr_selected_features.pkl"
)

print("✅ Logistic Regression (Feature Selection) model saved")
