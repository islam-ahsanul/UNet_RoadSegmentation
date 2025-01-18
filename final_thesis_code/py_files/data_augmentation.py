import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import (
    Compose, RandomBrightnessContrast, RandomRain, RandomFog, RandomSunFlare,
    HorizontalFlip, CoarseDropout
)

def load_training_data(path):
    """
    Load images and masks from the training directory.
    """
    images = sorted(glob(os.path.join(path, 'processed/train/images', '*.png')))
    masks = sorted(glob(os.path.join(path, 'processed/train/masks', '*.png')))
    return images, masks

def augment_training_data(images, masks, save_dir):
    """
    Apply augmentations to the training dataset and save the augmented dataset.
    """
    augmentations = Compose([
        RandomBrightnessContrast(p=0.2),
        RandomRain(p=0.2),
        RandomFog(p=0.2),
        RandomSunFlare(p=0.2),
        HorizontalFlip(p=0.5),
        CoarseDropout(p=0.2, max_holes=10, max_height=32, max_width=32),
    ])

    for img_path, mask_path in tqdm(zip(images, masks), total=len(images), desc="Augmenting training data"):
        img_name = os.path.basename(img_path).split('.')[0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is not None and mask is not None:
            augmented = augmentations(image=img, mask=mask)
            aug_img, aug_mask = augmented['image'], augmented['mask']

            # Save augmented images and masks
            cv2.imwrite(os.path.join(save_dir, 'images', f"{img_name}_aug.png"), aug_img)
            cv2.imwrite(os.path.join(save_dir, 'masks', f"{img_name}_aug.png"), aug_mask)

# Execute augmentation
dataset_path = '/path/to/dataset'
train_images, train_masks = load_training_data(dataset_path)

# Save augmented training data
augmented_dir = os.path.join(dataset_path, 'augmented_train')
os.makedirs(os.path.join(augmented_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(augmented_dir, 'masks'), exist_ok=True)

augment_training_data(train_images, train_masks, augmented_dir)
