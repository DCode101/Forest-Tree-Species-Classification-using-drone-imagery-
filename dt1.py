import matplotlib.pyplot as plt
import cv2
import os
import random

# Root dataset directory
data_dir = r"C:\Users\darsh\Downloads\dataset1\summer_dataset\train"

# List of target species (folder names should match or be close)
species_list = [
    "American chestnut", "black_cherry", "butternut",
    "northern_red_oak", "red_pine", "walnut",
    "white_oak", "white_pine"
]

species_images = []

# Gather one random image from each species folder
for species in species_list:
    species_path = os.path.join(data_dir, species)
    if os.path.isdir(species_path):
        images = [f for f in os.listdir(species_path) if f.lower().endswith(('.jpg', '.png'))]
        if images:
            selected_image = random.choice(images)
            image_path = os.path.join(species_path, selected_image)
            species_images.append((species, image_path))
        else:
            print(f"[!] No images found in folder: {species}")
    else:
        print(f"[!] Folder not found: {species}")

# Display images (2 rows, 8 columns: original and resized per species)
plt.figure(figsize=(20, 6))  # Wider figure for 8 species

for i, (label, img_path) in enumerate(species_images):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Row 1: Original image
    plt.subplot(2, len(species_images), i + 1)
    plt.imshow(img_rgb)
    plt.title(f"{label}\n(Original)", fontsize=8)
    plt.axis('off')

    # Row 2: Resized image
    plt.subplot(2, len(species_images), i + 1 + len(species_images))
    plt.imshow(img_resized)
    plt.title(f"{label}\n(224x224)", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
