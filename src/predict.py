import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# Paths
# =========================
BASE_DIR = r"D:\CropDiseaseDetection"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tomato_disease_cnn.keras"
)

# =========================
# Class names
# =========================
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# =========================
# Image settings
# =========================
IMAGE_SIZE = (128, 128)

# =========================
# Disease information
# =========================
DISEASE_INFO = {
    "Early Blight": {
        "description": "The leaf shows signs associated with Tomato Early Blight.",
        "advice": "Remove severely affected leaves and maintain good airflow around plants."
    },
    "Late Blight": {
        "description": "The leaf shows signs associated with Tomato Late Blight.",
        "advice": "Remove infected plant material and avoid prolonged leaf wetness."
    },
    "Healthy": {
        "description": "The leaf appears healthy based on the trained model.",
        "advice": "Continue regular monitoring and maintain proper plant care."
    }
}

# =========================
# Check image path
# =========================
if len(sys.argv) < 2:
    print("Usage:")
    print("python src\\predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("ERROR: Image not found.")
    print(image_path)
    sys.exit(1)

# =========================
# Load model
# =========================
print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Load and preprocess image
# =========================
print("Processing image...")

image = Image.open(image_path).convert("RGB")
image = image.resize(IMAGE_SIZE)

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

# =========================
# Prediction
# =========================
print("Making prediction...\n")

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = CLASS_NAMES[predicted_index]
confidence = predictions[0][predicted_index] * 100

# =========================
# Display result
# =========================
print("==============================")
print(" TOMATO DISEASE PREDICTION")
print("==============================")

print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")

print("\nClass probabilities:")

for class_name, probability in zip(
    CLASS_NAMES,
    predictions[0]
):
    print(f"{class_name:15s}: {probability * 100:.2f}%")

print("\nDescription:")
print(DISEASE_INFO[predicted_class]["description"])

print("\nRecommended Action:")
print(DISEASE_INFO[predicted_class]["advice"])

print("\n==============================")