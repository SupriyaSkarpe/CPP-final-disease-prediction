import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("dataset/early_diabetes.csv")

# Detect target column
target_col = [c for c in df.columns if c.lower() in ["class", "outcome", "target"]][0]

# Encode target
df[target_col] = df[target_col].map({
    "Positive": 1,
    "Negative": 0
})

X = df.drop(target_col, axis=1)
y = df[target_col]

# Identify categorical columns
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=False
)

model.fit(X_train, y_train, cat_features=cat_features)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("🔥 CatBoost Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
