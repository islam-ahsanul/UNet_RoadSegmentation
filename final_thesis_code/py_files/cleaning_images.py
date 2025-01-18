import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def filter_images_by_size(images, min_size=512):
    """
    Filters images by their size.
    Keeps only images with both height and width greater than min_size.
    """
    output = []
    for img in tqdm(images, total=len(images)):
        x = cv2.imread(img, cv2.IMREAD_COLOR)
        if x is not None:
            h, w, _ = x.shape
            if h > min_size and w > min_size:
                output.append(img)
    return output

def save_images(images, save_dir, size=(512, 512)):
    """
    Resizes and saves images in .png format to the specified directory.
    """
    for idx, path in enumerate(tqdm(images, total=len(images))):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        if x is not None:
            x = cv2.resize(x, size)  # Resize to model input size
            cv2.imwrite(f"{save_dir}/{idx + 1:04d}.png", x)  # Save as .png

# Load raw images
raw_images = glob("/path/to/road_images/*")  # Update to your dataset path
print("Initial images:", len(raw_images))

# Filter by size
output = filter_images_by_size(raw_images, min_size=512)
print("Filtered by size:", len(output))

# Save resized images
save_images(output, "/path/to/cleaned_images")  # Update output directory
