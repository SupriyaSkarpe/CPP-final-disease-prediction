# =========================================
# CATBOOST HEART DISEASE WITH FEATURE SELECTION
# Research-grade complete code
# =========================================

import pandas as pd
import joblib
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# =========================================
# 1. Load Dataset
# =========================================
df = pd.read_csv("dataset/heart.csv")
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

for col in cat_cols:
    df[col] = df[col].astype(str)

# =========================================
# 3. Split features and target
# =========================================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

cat_features = [X.columns.get_loc(col) for col in cat_cols]

# =========================================
# 4. INITIAL CatBoost (for feature importance)
# =========================================
base_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)

base_model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# =========================================
# 5. FEATURE SELECTION (Top features)
# =========================================
importances = base_model.get_feature_importance()
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Select top N features (SAFE RANGE: 8–12)
TOP_N = 10
selected_features = feature_importance_df.head(TOP_N)["feature"].tolist()

print("\n🔍 Selected Features:")
print(selected_features)

# =========================================
# 6. Reduce dataset to selected features
# =========================================
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

cat_features_sel = [
    X_train_sel.columns.get_loc(col)
    for col in selected_features
    if col in cat_cols
]

# =========================================
# 7. FINAL CatBoost Model (TUNED)
# =========================================
final_model = CatBoostClassifier(
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

final_model.fit(
    X_train_sel,
    y_train,
    cat_features=cat_features_sel,
    eval_set=(X_test_sel, y_test),
    use_best_model=True
)

# =========================================
# 8. Evaluation
# =========================================
y_pred = final_model.predict(X_test_sel)
y_prob = final_model.predict_proba(X_test_sel)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n🔥 CatBoost Accuracy (With Feature Selection):",
      round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(cm)

# =========================================
# 9. Save Model & Features
# =========================================
joblib.dump(final_model, "models/catboost_heart_fs_model.pkl")
joblib.dump(selected_features, "models/catboost_selected_features.pkl")

print("\n✅ CatBoost + Feature Selection model saved successfully!")
