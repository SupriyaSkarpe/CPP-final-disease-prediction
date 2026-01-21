import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# =========================
# 2. Encode categorical columns
# =========================
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# =========================
# 3. SIMPLE FEATURE SELECTION
# (drop weakest features for DT)
# =========================
drop_cols = ["FastingBS", "RestingECG"]  # weak for DT
df = df.drop(columns=drop_cols)

# =========================
# 4. Split features & target
# =========================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,      # IMPORTANT: improves DT generalization
    random_state=3,
    stratify=y
)

# =========================
# 5. Optimized Decision Tree
# =========================
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    min_samples_split=15,
    min_samples_leaf=6,
    max_features="sqrt",
    class_weight="balanced",
    random_state=3
)

# =========================
# 6. Train
# =========================
model.fit(X_train, y_train)

# =========================
# 7. Evaluate
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🔥 Decision Tree Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 8. Save model
# =========================
joblib.dump(model, "models/heart_dt_optimized.pkl")
joblib.dump(X.columns.tolist(), "models/heart_columns.pkl")


print("\nDecision Tree model saved successfully!")
