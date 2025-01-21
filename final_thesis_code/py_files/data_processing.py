import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_dataset(path):
    """
    Load images and masks from the specified directory.
    """
    images = sorted(glob(os.path.join(path, 'cleaned_images', '*.png')))
    masks = sorted(glob(os.path.join(path, 'masks', '*.png')))
    
    if not images or not masks:
        raise ValueError("No images or masks found. Check your dataset paths.")
    
    return images, masks

def split_dataset(images, masks, split=0.2):
    """
    Split dataset into training, validation, and test sets.
    """
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=split, random_state=42)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=split, random_state=42)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def create_dirs(path):
    """
    Create necessary directories if they don't exist.
    """
    os.makedirs(path, exist_ok=True)

def save_dataset(images, masks, save_dir):
    """
    Save images and masks to specified directory.
    """
    for x, y in tqdm(zip(images, masks), total=len(images), desc=f"Saving dataset in {save_dir}"):
        name = os.path.basename(x)
        img = cv2.imread(x, cv2.IMREAD_COLOR)
        mask = cv2.imread(y, cv2.IMREAD_COLOR)  # Load as color if using colored masks

        if img is None or mask is None:
            print(f"Warning: Issue loading {x} or {y}, skipping...")
            continue

        cv2.imwrite(os.path.join(save_dir, 'images', name), img)
        cv2.imwrite(os.path.join(save_dir, 'masks', name), mask)

# Execute processing
dataset_path = '/path/to/dataset'
images, masks = load_dataset(dataset_path)

# Split dataset
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_dataset(images, masks)

# Create directories for saving splits
processed_dir = os.path.join(dataset_path, 'processed')
for split in ['train', 'valid', 'test']:
    create_dirs(os.path.join(processed_dir, split, 'images'))
    create_dirs(os.path.join(processed_dir, split, 'masks'))

# Save datasets
save_dataset(train_x, train_y, os.path.join(processed_dir, 'train'))
save_dataset(valid_x, valid_y, os.path.join(processed_dir, 'valid'))
save_dataset(test_x, test_y, os.path.join(processed_dir, 'test'))
