import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# ===============================
# One-Hot Encoding (IMPORTANT)
# ===============================
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

# ===============================
# Split Features & Target
# ===============================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# Logistic Regression + Tuning
# ===============================
param_grid = {
    "C": [0.01, 0.1, 1, 5, 10, 50],
    "solver": ["liblinear", "lbfgs"],
    "penalty": ["l2"]
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

best_model = grid.best_estimator_

# ===============================
# Evaluation
# ===============================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("🔥 Logistic Regression Accuracy:", round(acc * 100, 2), "%")
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# Save Model
# ===============================
joblib.dump(best_model, "models/heart_lr_model.pkl")
joblib.dump(scaler, "models/heart_scaler.pkl")

print("✅ Logistic Regression model saved")
