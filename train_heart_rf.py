import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:/Users/HP/Documents/CPP-final-disease-prediction/datasets/heart.csv")

# Encode categorical columns
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", acc)

# Save model
joblib.dump(model, "C:/Users/HP/Documents/CPP-final-disease-prediction/models/heart_rf_model.pkl")
joblib.dump(le, "C:/Users/HP/Documents/CPP-final-disease-prediction/models/heart_label_encoder.pkl")

print("Model saved successfully!")
