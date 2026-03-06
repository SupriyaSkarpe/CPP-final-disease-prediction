import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.feature_selection import SelectFromModel
df = pd.read_csv("heart.csv")

# Check missing values
print(df.isnull().sum())

# If any missing values exist
df = df.dropna()

# Separate features & target
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 50, 100],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

lr = LogisticRegression(max_iter=1000)

grid = GridSearchCV(
    lr,
    param_grid,
    cv=10,
    scoring='accuracy'
)

grid.fit(X_train_scaled, y_train)

best_lr = grid.best_estimator_
print("Best Parameters:", grid.best_params_)
y_pred = best_lr.predict(X_test_scaled)

print("Accuracy (No Feature Selection):",
      accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    best_lr,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring='accuracy'
)

print("CV Accuracy Mean:", cv_scores.mean())
