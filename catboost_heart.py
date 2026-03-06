# =========================================
# CATBOOST HEART DISEASE PREDICTION
# Research-grade complete code
# =========================================

import pandas as pd
import joblib

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# =========================================
# 1. Load Dataset
# =========================================
df = pd.read_csv("dataset/heart.csv")   # <-- adjust path if needed
print("Dataset shape:", df.shape)

# =========================================
# 2. Identify categorical columns
# =========================================
cat_cols = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

# Convert categorical columns to string
for col in cat_cols:
    df[col] = df[col].astype(str)

# =========================================
# 3. Split features and target
# =========================================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Get categorical feature indices (required by CatBoost)
cat_features = [X.columns.get_loc(col) for col in cat_cols]

# =========================================
# 4. Build CatBoost Model (TUNED)
# =========================================
model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[
        1,
        (y == 0).sum() / (y == 1).sum()
    ],
    random_seed=42,
    verbose=100
)

# =========================================
# 5. Train Model
# =========================================
model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# =========================================
# 6. Evaluation
# =========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n🔥 CatBoost Accuracy:", round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(cm)

# =========================================
# 7. Save Model
# =========================================
joblib.dump(model, "models/catboost_heart_model.pkl")

print("\n✅ CatBoost model saved successfully!")
