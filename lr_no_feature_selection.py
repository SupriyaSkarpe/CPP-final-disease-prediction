import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
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
# 3. Encoding
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 6. Logistic Regression + tuning
# ===============================
param_grid = {
    "C": [0.01, 0.1, 1, 5, 10],
    "solver": ["liblinear", "lbfgs"]
}

lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

grid = GridSearchCV(
    lr,
    param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_lr = grid.best_estimator_

# ===============================
# 7. Evaluation
# ===============================
y_pred = best_lr.predict(X_test)
y_prob = best_lr.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("🔥 Logistic Regression Accuracy (No FS):", round(acc*100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# ===============================
# 8. Save model & scaler
# ===============================
joblib.dump(
    best_lr,
    "C:/Users/HP/Documents/CPP-final-disease-prediction/models/heart_lr_model.pkl"
)

joblib.dump(
    scaler,
    "C:/Users/HP/Documents/CPP-final-disease-prediction/models/heart_lr_scaler.pkl"
)

print("✅ Model and scaler saved successfully!")
