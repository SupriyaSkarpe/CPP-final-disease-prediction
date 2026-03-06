import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("dataset/early_diabetes.csv")
print("Dataset shape:", df.shape)

# =========================
# 2. Encode target column
# =========================
df["class"] = df["class"].map({"Positive": 1, "Negative": 0})

# =========================
# 3. Split X and y
# =========================
X = df.drop("class", axis=1)
y = df["class"]

# =========================
# 4. Manually specify categorical columns
# =========================
# All columns with Yes/No or Male/Female values
cat_columns = [
    "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis", "muscle stiffness",
    "Alopecia", "Obesity"
]

cat_features = [X.columns.get_loc(col) for col in cat_columns]
print("Categorical feature indexes:", cat_features)

# =========================
# 5. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 6. CatBoost model
# =========================
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_seed=42
)

# Train with categorical features
model.fit(
    X_train, y_train,
    cat_features=cat_features
)

# =========================
# 7. Evaluation
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n🔥 CatBoost Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 8. Save model
# =========================
joblib.dump(model, "models/early_diabetes_catboost.pkl")
joblib.dump(X.columns.tolist(), "models/early_diabetes_columns.pkl")

print("\n✅ CatBoost model saved successfully!")
