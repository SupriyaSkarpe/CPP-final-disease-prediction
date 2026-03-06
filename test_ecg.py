import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. Load trained model
model = tf.keras.models.load_model("model/ecg_model.h5")

# 2. Image settings (MATCH TRAINING)
IMG_SIZE = 224

# 3. Class names (MUST match training folder order)
CLASS_NAMES = ["Normal", "Arrhythmia"]

def test_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    
    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]

    # Results
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print("Image:", img_path)
    print("Predicted Class:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")
    print("-" * 40)

# 4. Test images
test_image("test_images/normal/normal_01.png")
test_image("test_images/abnormal/arrhythmia_01.png")
