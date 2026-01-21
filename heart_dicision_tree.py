import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("dataset/heart.csv")

# ===============================
# 2. One-hot encoding
# ===============================
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ===============================
# 3. Stratified split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. STRONG Decision Tree tuning
# ===============================
param_grid = {
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"]
}

dt = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

grid = GridSearchCV(
    dt,
    param_grid,
    cv=10,                # more stable
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_dt = grid.best_estimator_

# ===============================
# 5. Evaluation
# ===============================
y_pred = best_dt.predict(X_test)
y_prob = best_dt.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("🌳 Decision Tree Accuracy (MAX):", round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 6. Save model
# ===============================
joblib.dump(best_dt, "models/heart_decision_tree_best.pkl")
oblib.dump(X.columns.tolist(), "models/heart_columns.pkl")

print("✅ Best Decision Tree model saved")
