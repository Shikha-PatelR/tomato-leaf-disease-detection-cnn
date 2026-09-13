import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# Paths
# ==========================================

MODEL_PATH = r"D:\CropDiseaseDetection\models\tomato_disease_cnn.keras"
TEST_DIR = r"D:\CropDiseaseDetection\dataset\test"
RESULTS_DIR = r"D:\CropDiseaseDetection\results"

# ==========================================
# Configuration
# ==========================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# ==========================================
# Load Model
# ==========================================

print("=" * 50)
print("LOADING TRAINED CNN MODEL")
print("=" * 50)

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# Load Test Dataset
# ==========================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
print(class_names)

# ==========================================
# Evaluate Test Accuracy
# ==========================================

print("\n" + "=" * 50)
print("EVALUATING ON UNSEEN TEST DATA")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==========================================
# Generate Predictions
# ==========================================

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================
# Classification Report
# ==========================================

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# Save report
report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n"
    )
    file.write(report)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=15
)

plt.title("Tomato Disease CNN - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.show()

# ==========================================
# Final Message
# ==========================================

print("\n" + "=" * 50)
print("MODEL EVALUATION COMPLETED")
print("=" * 50)

print("\nSaved files:")
print(report_path)
print(cm_path)