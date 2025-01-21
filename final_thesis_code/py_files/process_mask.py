import numpy as np
import cv2
import json
import os

# Load annotation JSON
with open('/home/ahsan/University/Thesis/UNet_Directory/Datasets/annotated_json/annotations.json', 'r') as f:
    data = json.load(f)

# Paths for images and masks
img_dir = "/home/ahsan/University/Thesis/UNet_Directory/Datasets/cleaned_images_to_use/"
mask_dir = "/home/ahsan/University/Thesis/UNet_Directory/Datasets/mask_images"

# Create mask directory if not exists
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

images = data['images']
annotations = data['annotations']

# Define BGR colors for categories (OpenCV uses BGR format)
category_colors = {
    1: (79, 247, 211),  # #d3f74f in BGR (Greenish-Yellow for my-way)
    2: (247, 93, 79)    # #4f5df7 in BGR (Blue for other-way)
}

# Generate masks
for image in images:
    filename = image['file_name']
    height, width = image['height'], image['width']

    # Initialize mask with black background (3 channels for RGB)
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Process annotations for the image
    for annotation in [a for a in annotations if a['image_id'] == image['id']]:
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']

        # Assign BGR color for the category
        mask_color = category_colors.get(category_id, (0, 0, 0))  # Default to black

        # Draw polygons on the mask
        for points in segmentation:
            contours = []
            for i in range(0, len(points), 2):
                contours.append([int(points[i]), int(points[i + 1])])
            contours = np.array(contours, dtype=np.int32)
            cv2.drawContours(mask, [contours], -1, mask_color, -1)

    # Save the mask in .png format with RGB colors
    save_path = os.path.join(mask_dir, f"{filename.split('.')[0]}.png")
    cv2.imwrite(save_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"Saved mask: {save_path}")