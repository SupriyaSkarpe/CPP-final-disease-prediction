import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
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
    raise Exception("Target column not found")


# =====================================
# CONVERT TARGET
# =====================================
df[target_col] = df[target_col].map({
    "Positive": 1,
    "Negative": 0
})


# =====================================
# LABEL ENCODE CATEGORICAL FEATURES
# =====================================
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


# =====================================
# SPLIT FEATURES AND TARGET
# =====================================
X = df.drop(target_col, axis=1)
y = df[target_col]


# =====================================
# FEATURE SCALING
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =====================================
# TRAIN TEST SPLIT (60:40)
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.4,     # 40% test data
    stratify=y,
    random_state=42
)


# =====================================
# TRAIN MODEL
# =====================================
model = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)

model.fit(X_train, y_train)


# =====================================
# PREDICTION
# =====================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]


# =====================================
# EVALUATION
# =====================================
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("🔥 KNN Accuracy:", round(accuracy*100,2), "%")
print("ROC-AUC:", round(roc_auc,4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =====================================
# SAVE MODEL FOR API
# =====================================
joblib.dump(model, "models/diabetes_knn_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")
joblib.dump(X.columns, "models/diabetes_columns.pkl")

print("\n✅ Model saved successfully")