import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/heart.csv"
)


df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True]
}

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_


y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🔥 Random Forest Accuracy:", round(acc * 100, 2), "%")
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(best_model, "models/heart_rf_model.pkl")
joblib.dump(X.columns.tolist(), "models/heart_columns.pkl")


print("✅ Random Forest model saved")
