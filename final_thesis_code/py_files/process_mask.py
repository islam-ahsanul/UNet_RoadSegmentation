import numpy as np
import cv2
import json
import os

# Load annotation JSON
annotation_file = '/path/to/annotations.json'
with open(annotation_file, 'r') as f:
    data = json.load(f)

# Paths for images and masks
img_dir = "/path/to/cleaned_images_to_use/"
mask_dir = "/path/to/mask_images"

# Create mask directory if not exists
os.makedirs(mask_dir, exist_ok=True)

# Define grayscale values for categories (1 = My Way, 2 = Other Way)
category_colors = {
    1: 128,  # My Way (gray)
    2: 255   # Other Way (white)
}

# Generate masks
for image in data['images']:
    filename = image['file_name']
    height, width = image['height'], image['width']

    # Initialize mask with black background (grayscale)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process annotations for the image
    for annotation in [a for a in data['annotations'] if a['image_id'] == image['id']]:
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']

        # Assign grayscale value for the category
        mask_color = category_colors.get(category_id, 0)  # Default to black (0)

        # Draw polygons on the mask
        for points in segmentation:
            contours = []
            for i in range(0, len(points), 2):
                contours.append([int(points[i]), int(points[i + 1])])
            contours = np.array(contours, dtype=np.int32)
            cv2.fillPoly(mask, [contours], mask_color)

    # Save the mask in .png format with grayscale values
    save_path = os.path.join(mask_dir, f"{filename.split('.')[0]}.png")
    cv2.imwrite(save_path, mask)

    print(f"Saved mask: {save_path}")

print("Mask generation completed successfully!")
