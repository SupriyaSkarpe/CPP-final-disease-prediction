import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(
    "C:/Users/HP/Documents/CPP-final-disease-prediction/dataset/clevelant.csv",
    encoding="latin-1",
    header=None
)

# Assign column names
df.columns = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak',
    'slope','ca','thal','target'
]

# Replace ? with NaN and drop
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert to numeric
df = df.astype(float)

# Split
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
