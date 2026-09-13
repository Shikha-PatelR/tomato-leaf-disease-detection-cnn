import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# Paths
# =========================
BASE_DIR = r"D:\CropDiseaseDetection"

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tomato_disease_mobilenetv2.keras"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =========================
# Configuration
# =========================
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# =========================
# Load test dataset
# =========================
print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:", class_names)
print("Number of classes:", len(class_names))

# =========================
# Load model
# =========================
print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Evaluate
# =========================
print("\nEvaluating MobileNetV2...")

loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\n==============================")
print("MOBILENETV2 TEST RESULTS")
print("==============================")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# =========================
# Predictions
# =========================
y_true = np.concatenate([
    y.numpy() for x, y in test_ds
])

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

# =========================
# Classification Report
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:")
print(report)

report_path = os.path.join(
    RESULTS_DIR,
    "mobilenet_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("MobileNetV2 Classification Report\n")
    f.write("================================\n\n")
    f.write(report)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    values_format="d",
    xticks_rotation=45
)

plt.title("MobileNetV2 Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "mobilenet_confusion_matrix.png"
)

plt.savefig(cm_path)
plt.close()

print("\nResults saved:")
print(report_path)
print(cm_path)

print("\n==============================")
print("MOBILENETV2 EVALUATION COMPLETE")
print("==============================")