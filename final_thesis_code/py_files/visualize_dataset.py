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
    mask_path = os.path.join(masks_dir, name.replace(".jpg", ".png").replace(".jpeg", ".png"))

    # Load the image and mask
    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    y = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if x is not None and y is not None:
        # Create a separator line
        line = np.ones((x.shape[0], 10, 3)) * 255

        # Apply a colormap to the mask for visualization
        y_colored = cv2.applyColorMap((y * 50).astype(np.uint8), cv2.COLORMAP_JET)

        # Concatenate the image and the mask
        cat_img = np.concatenate([x, line, y_colored], axis=1)

        # Save the visualized image
        save_path = os.path.join(output_dir, name.replace(".jpg", ".png").replace(".jpeg", ".png"))
        cv2.imwrite(save_path, cat_img)
