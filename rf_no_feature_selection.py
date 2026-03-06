import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# ===============================
# 2. Data Cleaning
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
# 5. Random Forest + Tuning
# ===============================
param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [None, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

# ===============================
# 6. Evaluation
# ===============================
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("🔥 Random Forest Accuracy (No FS):", round(acc*100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save model
# ===============================
joblib.dump(best_rf, "models/heart_rf_best.pkl")
print("✅ Random Forest model saved")
