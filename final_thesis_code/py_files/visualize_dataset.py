import os
import numpy as np
import cv2
from tqdm import tqdm

# Directory paths
cleaned_images_dir = "/path/to/cleaned_images"
masks_dir = "/path/to/masks"
output_dir = "/path/to/visualized_dataset"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load all image names
image_names = sorted(os.listdir(cleaned_images_dir))
print(f"Total images: {len(image_names)}")

# Visualize images and masks side by side
for name in tqdm(image_names, desc="Visualizing dataset"):
    image_path = os.path.join(cleaned_images_dir, name)
    mask_name = os.path.splitext(name)[0] + ".png"  # Convert to .png for masks
    mask_path = os.path.join(masks_dir, mask_name)

    # Check if the mask exists before processing
    if not os.path.exists(mask_path):
        print(f"Mask not found for {name}, skipping...")
        continue

    # Load the image and mask in COLOR mode
    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    y = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # Load mask in color

    if x is None or y is None:
        print(f"Error loading {name} or corresponding mask")
        continue

    # Create a separator line
    line = np.ones((x.shape[0], 10, 3), dtype=np.uint8) * 255

    # Concatenate the original image with the colored mask
    cat_img = np.concatenate([x, line, y], axis=1)

    # Save the visualized image
    save_path = os.path.join(output_dir, mask_name)
    cv2.imwrite(save_path, cat_img)

print("Visualization complete.")
