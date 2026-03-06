import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# =====================================
# LOAD DATASET
# =====================================
df = pd.read_csv("dataset/early_diabetes.csv")


# =====================================
# FIND TARGET COLUMN
# =====================================
target_col = None

for col in df.columns:
    if col.lower() in ["class", "outcome", "target"]:
        target_col = col
        break

if target_col is None:
    raise Exception("❌ Target column not found")


# =====================================
# CONVERT TARGET VALUES
# =====================================
df[target_col] = df[target_col].map({
    "Positive": 1,
    "Negative": 0,
    "Yes": 1,
    "No": 0
})


# =====================================
# ONE HOT ENCODING
# =====================================
df = pd.get_dummies(df, drop_first=True)


# =====================================
# SPLIT FEATURES AND LABEL
# =====================================
X = df.drop(target_col, axis=1)
y = df[target_col]


# =====================================
# TRAIN TEST SPLIT (70:30)
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,      # 30% testing
    stratify=y,
    random_state=42
)


# =====================================
# TRAIN MODEL
# =====================================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# =====================================
# PREDICTION
# =====================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =====================================
# PERFORMANCE
# =====================================
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("🔥 Random Forest Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC Score:", round(roc_auc, 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =====================================
# SAVE MODEL FOR API
# =====================================
joblib.dump(model, "models/diabetes_rf_model.pkl")
joblib.dump(X.columns, "models/diabetes_columns.pkl")

print("\n✅ Model saved successfully")