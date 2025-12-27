import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv("dataset/diabetes.csv")

# ===============================
# 2️⃣ Features & Target
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# 3️⃣ Train-Test Split (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4️⃣ Pipeline (Scaler + Model)
# ===============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

# ===============================
# 5️⃣ Hyperparameter Grid
# ===============================
param_grid = {
    "rf__n_estimators": [150, 200, 300],
    "rf__max_depth": [6, 8, 10, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4]
}

# ===============================
# 6️⃣ Grid Search with CV
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="recall",   # IMPORTANT: focus on diabetes detection
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ===============================
# 7️⃣ Best Model Evaluation
# ===============================
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("✅ Best Parameters:", grid.best_params_)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 8️⃣ Save Model
# ===============================
joblib.dump(best_model, "models/diabetes_rf_model.pkl")

print("\n💾 Diabetes Random Forest model saved successfully!")
