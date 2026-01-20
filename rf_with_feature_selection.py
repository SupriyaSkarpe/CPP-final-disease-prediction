import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ===============================
# 3. Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. Initial RF (for feature importance)
# ===============================
rf_initial = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

rf_initial.fit(X_train, y_train)

importances = rf_initial.feature_importances_

# ===============================
# 5. Select top features
# ===============================
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

selected_features = importance_df.head(12)["feature"].tolist()

print("Selected Features:")
print(selected_features)

# ===============================
# 6. Train final RF on selected features
# ===============================
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

rf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    class_weight="balanced"
)

rf_final.fit(X_train_sel, y_train)

# ===============================
# 7. Evaluation
# ===============================
y_pred = rf_final.predict(X_test_sel)
y_prob = rf_final.predict_proba(X_test_sel)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🔥 Random Forest Accuracy (With FS):", round(acc*100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Save model
# ===============================
joblib.dump(rf_final, "models/heart_rf_fs.pkl")
print("✅ Random Forest + Feature Selection model saved")
