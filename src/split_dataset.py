import os
import shutil
import random

# ==============================
# Paths
# ==============================

RAW_DIR = r"D:\CropDiseaseDetection\dataset\raw"
DATASET_DIR = r"D:\CropDiseaseDetection\dataset"

# ==============================
# Split ratios
# ==============================

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducible split
random.seed(42)

# ==============================
# Class names
# ==============================

classes = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

# ==============================
# Create directories
# ==============================

for split in ["train", "validation", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(DATASET_DIR, split, class_name),
            exist_ok=True
        )

# ==============================
# Split images
# ==============================

print("=" * 50)
print("DATASET SPLITTING")
print("=" * 50)

for class_name in classes:

    source_dir = os.path.join(RAW_DIR, class_name)

    images = [
        file for file in os.listdir(source_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    print(f"\n{class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

    for split_name, split_images in splits.items():

        destination_dir = os.path.join(
            DATASET_DIR,
            split_name,
            class_name
        )

        for image in split_images:

            source_path = os.path.join(
                source_dir,
                image
            )

            destination_path = os.path.join(
                destination_dir,
                image
            )

            shutil.copy2(
                source_path,
                destination_path
            )

# ==============================
# Final message
# ==============================

print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETED")
print("=" * 50)