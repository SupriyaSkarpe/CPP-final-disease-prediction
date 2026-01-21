import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("dataset/heart.csv")
print("Dataset shape:", df.shape)

# ===============================
# 2. One-hot encoding
# ===============================
df = pd.get_dummies(
    df,
    columns=['Sex', 'ChestPainType', 'RestingECG',
             'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)

# ===============================
# 3. Split X & y
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
# 4. Feature Scaling (CRITICAL)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. Neural Network Model
# ===============================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 6. Early Stopping
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# ===============================
# 7. Train model
# ===============================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# 8. Evaluation
# ===============================
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🧠 Neural Network Accuracy:", round(acc * 100, 2), "%")
print("ROC-AUC:", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9. Save model
# ===============================
model.save("models/heart_ann_model.h5")
joblib.dump(scaler, "models/heart_ann_scaler.pkl")
joblib.dump(X.columns.tolist(), "models/heart_columns.pkl")


print("✅ Neural Network model & scaler saved successfully!")
