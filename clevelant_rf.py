import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Load Cleveland dataset (WITH header)
# ==============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart_cleveland.csv",
    encoding="latin-1"
)

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# ==============================
# 2. Data cleaning
# ==============================
df.replace("?", np.nan, inplace=True)

# Convert numeric columns safely
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.fillna(df.median(), inplace=True)

# ==============================
# 3. Convert target to binary
# ==============================
df["condition"] = df["condition"].apply(lambda x: 1 if x > 0 else 0)

# ==============================
# 4. One-hot encoding
# ==============================
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ==============================
# 5. Split X & y
# ==============================
X = df.drop("condition", axis=1)
y = df["condition"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# 6. Random Forest
# ==============================
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 7. Evaluation
# ==============================
y_pred = model.predict(X_test)

print("\n🔥 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 8. Save model
# ==============================
joblib.dump(model, "models/cleveland_rf_model.pkl")
print("\n✅ Model saved successfully")
