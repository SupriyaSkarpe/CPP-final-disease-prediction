import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("dataset/diabetes_pima.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# =========================
# 2. Replace invalid zeros
# =========================
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# 3. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 4. Pipeline (CRITICAL)
# =========================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),  # REQUIRED for SVM
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",  # handles imbalance
        random_state=42
    ))
])

# =========================
# 5. Hyperparameter tuning
# =========================
param_grid = {
    "svm__C": [0.5, 1, 5, 10],
    "svm__gamma": [0.01, 0.05, 0.1, "scale"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# =========================
# 6. Prediction
# =========================
y_pred = best_model.predict(X_test)

# =========================
# 7. Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)
print("\n🔥 SVM Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 8. Save model
# =========================
joblib.dump(best_model, "models/pima_svm_model.pkl")

print("\n✅ SVM model saved successfully!")
