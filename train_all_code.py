import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# ===============================
# 2. Cleaning
# ===============================
df = df.drop_duplicates()

df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ===============================
# 3. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. Scaling (for LR, KNN, SVM)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. MODELS
# ===============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000, class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=11,
        weights="distance"
    ),

    "SVM (RBF)": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced"
    )
}

# ===============================
# 6. Training & Evaluation
# ===============================
results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN", "SVM (RBF)"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append((name, round(acc * 100, 2)))

    print("\n", "="*50)
    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, y_pred))

# ===============================
# 7. Final comparison
# ===============================
print("\nFINAL ACCURACY COMPARISON")
for r in results:
    print(r[0], "→", r[1], "%")

# Save best model (SVM expected)
joblib.dump(models["SVM (RBF)"], "models/best_model_svm.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Best model and scaler saved")
