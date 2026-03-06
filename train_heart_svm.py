import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# =========================
# 2. Encode categorical columns
# =========================
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# 3. Features & target
# =========================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================
# 4. Stratified split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. Scaling
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 6. SVM + GridSearch with wider param range
# =========================
param_grid = {
    "C": [0.1, 1, 10, 50, 100, 200],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5],
    "kernel": ["rbf", "poly", "sigmoid"]
}

svm = SVC(probability=True, class_weight="balanced", random_state=42)

grid = GridSearchCV(
    svm,
    param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# =========================
# 7. Train
# =========================
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# =========================
# 8. Evaluate
# =========================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("🔥 SVM Accuracy:", round(acc * 100, 2), "%")
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 9. Save model
# =========================
joblib.dump(best_model, "models/heart_svm_best.pkl")
joblib.dump(scaler, "models/heart_scaler.pkl")
joblib.dump(label_encoders, "models/heart_label_encoders.pkl")

print("\n✅ SVM model saved successfully!")
