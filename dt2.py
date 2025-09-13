import os
import json
import cv2
import matplotlib.pyplot as plt
import random

# --- Paths ---
data_dir = r"C:\Users\darsh\Downloads\dataset2\train"
json_file = os.path.join(data_dir, "_annotations.coco.json")

# Target species
species_list = ["spruce", "fir", "pine", "trembling_aspen"]

# Load JSON
with open(json_file, "r") as f:
    coco = json.load(f)

cat_map = {cat["id"]: cat["name"].lower() for cat in coco["categories"]}
image_map = {img["id"]: img for img in coco["images"]}

# Collect sample images
species_images = {}
for ann in coco["annotations"]:
    cat_name = cat_map.get(ann["category_id"], "").lower()
    if cat_name in species_list and cat_name not in species_images:
        img_info = image_map[ann["image_id"]]
        img_path = os.path.join(data_dir, img_info["file_name"])
        if os.path.exists(img_path):
            species_images[cat_name] = img_path
    if len(species_images) == len(species_list):
        break

# Display before vs after
plt.figure(figsize=(12, 6))
for i, (species, img_path) in enumerate(species_images.items()):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Original
    plt.subplot(2, len(species_images), i + 1)
    plt.imshow(img_rgb)
    plt.title(f"{species}\n(Original)", fontsize=8)
    plt.axis("off")

    # Resized
    plt.subplot(2, len(species_images), i + 1 + len(species_images))
    plt.imshow(img_resized)
    plt.title(f"{species}\n(224x224)", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()
