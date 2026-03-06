import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH = 32

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "ecg_dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    "ecg_dataset/test",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

# 🔥 THIS LINE GIVES FINAL ACCURACY
loss, accuracy = model.evaluate(test_data)
print("✅ Test Accuracy:", accuracy * 100, "%")

model.save("models/ecg_cnn_model.h5")
