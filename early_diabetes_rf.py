import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("dataset/early_diabetes.csv")

# 🔍 Detect target column safely
target_col = None
for col in df.columns:
    if col.lower() in ["class", "outcome", "target"]:
        target_col = col
        break

if target_col is None:
    raise Exception("❌ Target column not found")

# Encode target (Positive / Negative → 1 / 0)
df[target_col] = df[target_col].map({
    "Positive": 1,
    "Negative": 0,
    "Yes": 1,
    "No": 0
})

# One-hot encode features
df = pd.get_dummies(df, drop_first=True)

X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Random Forest model (tuned)
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Results
print("🔥 Random Forest Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
