import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_csv("dataset/heart.csv")

df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("🔥 Gradient Boosting Accuracy:",
      round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/heart_gb.pkl")
