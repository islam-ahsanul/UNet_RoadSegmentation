import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations.augmentations.transforms import (
    RandomBrightnessContrast, RandomRain, RandomFog, RandomSunFlare,
    CoarseDropout
)
from albumentations import Compose

def load_training_data(path):
    """
    Load images and masks from the training directory.
    """
    images = sorted(glob(os.path.join(path, 'processed/train/images', '*.png')))
    masks = sorted(glob(os.path.join(path, 'processed/train/masks', '*.png')))
    return images, masks

def augment_training_data(images, masks, save_dir):
    """
    Apply augmentations to the training dataset and save the augmented dataset separately,
    ensuring that the mask values are preserved.
    """

    augmentation_list = [
        ("brightness_contrast", RandomBrightnessContrast(p=1.0)),
        ("rain", RandomRain(p=1.0)),
        ("fog", RandomFog(p=1.0)),
        ("sunflare", RandomSunFlare(p=1.0)),
        ("coarsedropout", CoarseDropout(p=1.0, max_holes=10, max_height=32, max_width=32))
    ]

    for img_path, mask_path in tqdm(zip(images, masks), total=len(images), desc="Augmenting training data"):
        img_name = os.path.basename(img_path).split('.')[0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if img is not None and mask is not None:
            for aug_name, aug in augmentation_list:
                augmented = Compose([aug], additional_targets={'mask': 'image'})(image=img, mask=mask)
                aug_img, aug_mask = augmented['image'], augmented['mask']

                # Save augmented images and masks separately with original mask colors
                cv2.imwrite(os.path.join(save_dir, 'images', f"{img_name}_{aug_name}.png"), aug_img)
                cv2.imwrite(os.path.join(save_dir, 'masks', f"{img_name}_{aug_name}.png"), aug_mask)

# Execute augmentation
dataset_path = '/path/to/dataset'
train_images, train_masks = load_training_data(dataset_path)

# Save augmented training data
augmented_dir = os.path.join(dataset_path, 'augmented_train')
os.makedirs(os.path.join(augmented_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(augmented_dir, 'masks'), exist_ok=True)

augment_training_data(train_images, train_masks, augmented_dir)
