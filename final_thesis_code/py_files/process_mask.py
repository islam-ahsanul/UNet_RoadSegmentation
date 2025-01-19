import numpy as np
import cv2
import json

# Load annotation JSON
with open('/home/ahsan/University/Thesis/UNet_Directory/Datasets/annotated_json/annotations.json', 'r') as f:
    data = json.load(f)

# Paths for images and masks
img_dir = "/home/ahsan/University/Thesis/UNet_Directory/Datasets/cleaned_images/"  # Directory containing cleaned images
mask_dir = "/home/ahsan/University/Thesis/UNet_Directory/Datasets/mask_images"         # Directory to save generated masks

images = data['images']
annotations = data['annotations']
categories = {cat['id']: cat['name'] for cat in data['categories']}

# Generate masks
for image in images:
    filename = image['file_name']
    height, width = image['height'], image['width']

    # Initialize mask with background (0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process annotations for the image
    for annotation in [a for a in annotations if a['image_id'] == image['id']]:
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']

        # Draw polygons on the mask
        for points in segmentation:
            contours = []
            for i in range(0, len(points), 2):
                contours.append([points[i], points[i + 1]])
            contours = np.array(contours, dtype=np.int32)
            cv2.drawContours(mask, [contours], -1, category_id, -1)

    # Save the mask in .png format
    cv2.imwrite(f"{mask_dir}/{filename.split('.')[0]}.png", mask)
