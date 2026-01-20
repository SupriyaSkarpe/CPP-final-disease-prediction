import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("dataset/heart.csv")
df = pd.get_dummies(df, drop_first=True)

target = "HeartDisease"
X = df.drop(target, axis=1)
y = df[target]

# ===============================
# 2. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 3. Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 4. Feature Selection using XGBoost
# ===============================
selector_model = XGBClassifier(
    n_estimators=200,
    random_state=42,
    eval_metric="logloss"
)

selector = SelectFromModel(
    selector_model,
    threshold="median"   # keeps top 50% important features
)

selector.fit(X_train_scaled, y_train)

X_train_fs = selector.transform(X_train_scaled)
X_test_fs = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print("🔍 Selected Features:")
print(list(selected_features))

# ===============================
# 5. Final XGBoost Model
# ===============================
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_fs, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_pred = model.predict(X_test_fs)
y_prob = model.predict_proba(X_test_fs)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🔥 XGBoost + Feature Selection Accuracy:", round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save models
# ===============================
joblib.dump(model, "models/heart_xgb_fs_model.pkl")
joblib.dump(scaler, "models/heart_xgb_scaler.pkl")
joblib.dump(selector, "models/heart_xgb_selector.pkl")

print("✅ Model, scaler & selector saved")
