import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)

# ===============================
# 2. Cleaning
# ===============================
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ===============================
# 3. One-hot encoding
# ===============================
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

# ===============================
# 4. Split
# ===============================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 5. Scaling (MANDATORY for KNN)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 6. KNN + Hyperparameter tuning
# ===============================
param_grid = {
    "n_neighbors": [3,5,7,9,11,13],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

knn = KNeighborsClassifier()

grid = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_knn = grid.best_estimator_

# ===============================
# 7. Evaluation
# ===============================
y_pred = best_knn.predict(X_test)
y_prob = best_knn.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("🔥 KNN Accuracy (No FS):", round(acc*100,2), "%")
print("ROC-AUC:", round(roc,4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Save
# ===============================
joblib.dump(best_knn, "models/heart_knn_model.pkl")
joblib.dump(scaler, "models/heart_knn_scaler.pkl")

print("✅ KNN model saved")
