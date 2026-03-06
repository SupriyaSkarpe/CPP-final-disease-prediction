import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("dataset/early_diabetes.csv")

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# 2. Encode categorical values
# =========================
# Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Yes/No columns
yes_no_cols = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity"
]

for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Target
df["class"] = df["class"].map({"Positive": 1, "Negative": 0})

# =========================
# 3. Split X and y
# =========================
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 4. Train XGBoost model
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 5. Evaluation
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🔥 XGBoost Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 6. Save model
# =========================
joblib.dump(model, "models/early_diabetes_xgb.pkl")
joblib.dump(X.columns.tolist(), "models/early_diabetes_columns.pkl")

print("\n✅ Model saved successfully!")
