import os
from PIL import Image
import matplotlib.pyplot as plt

# ==============================
# Dataset Path
# ==============================

DATASET_PATH = r"D:\CropDiseaseDetection\dataset\raw"

# ==============================
# Class Names
# ==============================

classes = {
    "Tomato___healthy": "Healthy",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight"
}

print("=" * 50)
print("TOMATO LEAF DISEASE DATASET - EDA")
print("=" * 50)

# ==============================
# 1. Count Images
# ==============================

counts = {}

for folder in classes:
    folder_path = os.path.join(DATASET_PATH, folder)

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    counts[classes[folder]] = len(image_files)

print("\nImage Counts:")
for class_name, count in counts.items():
    print(f"{class_name}: {count}")

print(f"\nTotal Images: {sum(counts.values())}")

# ==============================
# 2. Check Image Sizes
# ==============================

print("\nChecking image dimensions...")

sizes = []

for folder in classes:
    folder_path = os.path.join(DATASET_PATH, folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)

            try:
                with Image.open(image_path) as img:
                    sizes.append(img.size)
            except Exception as e:
                print(f"Problem with: {image_path}")
                print(e)

print(f"Number of images checked: {len(sizes)}")
print(f"First image size: {sizes[0]}")
print(f"Unique image sizes: {set(sizes)}")

# ==============================
# 3. Display Sample Images
# ==============================

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for row, folder in enumerate(classes):

    folder_path = os.path.join(DATASET_PATH, folder)

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for col in range(3):

        image_path = os.path.join(
            folder_path,
            image_files[col]
        )

        image = Image.open(image_path)

        axes[row, col].imshow(image)
        axes[row, col].set_title(classes[folder])
        axes[row, col].axis("off")

plt.suptitle(
    "Sample Tomato Leaf Images",
    fontsize=16
)

plt.tight_layout()

# Save figure
os.makedirs(
    r"D:\CropDiseaseDetection\results",
    exist_ok=True
)

plt.savefig(
    r"D:\CropDiseaseDetection\results\sample_images.png",
    dpi=150
)

plt.show()

# ==============================
# 4. Class Distribution
# ==============================

plt.figure(figsize=(8, 5))

plt.bar(
    counts.keys(),
    counts.values()
)

plt.title("Tomato Disease Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    r"D:\CropDiseaseDetection\results\class_distribution.png",
    dpi=150
)

plt.show()

print("\nEDA completed successfully!")
print("Graphs saved inside:")
print(r"D:\CropDiseaseDetection\results")