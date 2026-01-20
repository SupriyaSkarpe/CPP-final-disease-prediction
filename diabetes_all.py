import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("dataset/diabetes_pima.csv")
print("Dataset shape:", df.shape)

# ===============================
# 2. Data Cleaning (CRITICAL)
# ===============================
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4. Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. Models Dictionary
# ===============================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "SVM": SVC(kernel="rbf", C=2, gamma="scale", probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )
}

# ===============================
# 6. Train & Evaluate
# ===============================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n🔥 {name}")
    print("Accuracy:", round(acc*100, 2), "%")
    print("ROC-AUC:", round(roc, 4))
    print(classification_report(y_test, y_pred))

    results.append([name, acc, roc])

# ===============================
# 7. Stacking Ensemble (BEST)
# ===============================
base_models = [
    ("rf", models["Random Forest"]),
    ("gb", models["Gradient Boosting"]),
    ("xgb", models["XGBoost"])
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=GradientBoostingClassifier(),
    cv=5
)

stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)

acc_stack = accuracy_score(y_test, y_pred_stack)
roc_stack = roc_auc_score(y_test, stacking.predict_proba(X_test)[:, 1])

print("\n🔥 Stacking Ensemble")
print("Accuracy:", round(acc_stack*100, 2), "%")
print("ROC-AUC:", round(roc_stack, 4))
print(classification_report(y_test, y_pred_stack))
