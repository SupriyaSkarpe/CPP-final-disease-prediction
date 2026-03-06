import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv")

# =========================
# 2. Encode categorical columns
# =========================
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# 3. Features & target
# =========================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================
# 4. Feature selection using RandomForest
# =========================
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X, y)

selector = SelectFromModel(rf, prefit=True, threshold="median")
X_selected = selector.transform(X)
selected_features = X.columns[selector.get_support()]
print("Selected Features:", list(selected_features))

# =========================
# 5. Stratified split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 6. Scaling (optional for XGBoost, but safe)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. XGBoost + GridSearch
# =========================
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

grid = GridSearchCV(
    xgb_clf,
    param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# =========================
# 8. Train
# =========================
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# =========================
# 9. Evaluate
# =========================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("🔥 XGBoost Accuracy after Feature Selection:", round(acc * 100, 2), "%")
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 10. Save model
# =========================
joblib.dump(best_model, "models/heart_xgb_best_fs.pkl")
joblib.dump(scaler, "models/heart_scaler_fs.pkl")
joblib.dump(label_encoders, "models/heart_label_encoders_fs.pkl")
joblib.dump(selected_features, "models/heart_selected_features.pkl")

print("\n✅ XGBoost model with Feature Selection saved successfully!")
