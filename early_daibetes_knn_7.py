import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset/early_diabetes.csv")


# =========================
# CONVERT TARGET COLUMN
# =========================
df["class"] = df["class"].map({
    "Positive": 1,
    "Negative": 0
})


# =========================
# CONVERT YES/NO FEATURES
# =========================
binary_cols = [
"Polyuria",
"Polydipsia",
"sudden weight loss",
"weakness",
"Polyphagia",
"visual blurring"
]

for col in binary_cols:
    df[col] = df[col].map({"Yes":1,"No":0})

df["Gender"] = df["Gender"].map({"Male":1,"Female":0})


# =========================
# SELECT IMPORTANT FEATURES
# =========================
features = [
"Age",
"Polyuria",
"Polydipsia",
"sudden weight loss",
"weakness",
"Polyphagia",
"visual blurring"
]

X = df[features]
y = df["class"]


# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# TRAIN TEST SPLIT (60:40)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.4,   # 40% testing
    stratify=y,
    random_state=42
)


# =========================
# TRAIN MODEL
# =========================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]


# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("🔥 KNN Accuracy:", round(accuracy*100,2), "%")
print("ROC-AUC:", round(roc,4))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))


# =========================
# SAVE MODEL
# =========================
joblib.dump(model,"models/diabetes_knn_model.pkl")
joblib.dump(scaler,"models/diabetes_scaler.pkl")
joblib.dump(features,"models/diabetes_features.pkl")

print("\n✅ Model saved successfully")