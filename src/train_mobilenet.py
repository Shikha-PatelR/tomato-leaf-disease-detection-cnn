import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# =========================
# Paths
# =========================
BASE_DIR = r"D:\CropDiseaseDetection"

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Configuration
# =========================
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# =========================
# Load datasets
# =========================
print("Loading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = train_ds.class_names

print("\nClasses:", class_names)
print("Number of classes:", len(class_names))

# =========================
# Performance optimization
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# =========================
# Data augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# =========================
# MobileNetV2 base model
# =========================
print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# Build model
# =========================
inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# =========================
# Compile
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# =========================
# Train
# =========================
print("\n==============================")
print("STARTING MOBILENETV2 TRAINING")
print("==============================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =========================
# Save model
# =========================
model_path = os.path.join(
    MODEL_DIR,
    "tomato_disease_mobilenetv2.keras"
)

model.save(model_path)

print("\nModel saved to:")
print(model_path)

# =========================
# Save accuracy graph
# =========================
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

accuracy_path = os.path.join(
    RESULTS_DIR,
    "mobilenet_accuracy.png"
)

plt.savefig(accuracy_path)
plt.close()

# =========================
# Save loss graph
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("MobileNetV2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

loss_path = os.path.join(
    RESULTS_DIR,
    "mobilenet_loss.png"
)

plt.savefig(loss_path)
plt.close()

print("\nGraphs saved:")
print(accuracy_path)
print(loss_path)

print("\n==============================")
print("MOBILENETV2 TRAINING COMPLETE")
print("==============================")