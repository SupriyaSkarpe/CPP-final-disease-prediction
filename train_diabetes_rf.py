import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("dataset/diabetes_pima.csv")
print("Dataset shape:", df.shape)

# =========================
# 2. Handle zero values (medically invalid)
# =========================
zero_cols = [
    "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI"
]

for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# =========================
# 3. Split X and y
# =========================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 5. Random Forest + GridSearch
# =========================
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True]
}

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

grid = GridSearchCV(
    rf,
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
print("\n🔥 Random Forest Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 8. Cross-validation score
# =========================
cv_score = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation Accuracy:", cv_score.mean())

# =========================
# 9. Save model
# =========================
joblib.dump(best_model, "models/pima_rf_model.pkl")

print("\n✅ Random Forest model saved successfully!")
