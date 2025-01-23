import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import Compose
from albumentations.augmentations.transforms import (
    RandomBrightnessContrast, RandomRain, RandomFog, RandomSunFlare,
    CoarseDropout, HorizontalFlip, VerticalFlip
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
    Apply augmentations to the training dataset and save the augmented dataset separately,
    ensuring that the mask values are preserved.
    """

    augmentation_list = [
        ("brightness_contrast", RandomBrightnessContrast(p=1.0)),
        ("rain", RandomRain(p=1.0)),
        ("fog", RandomFog(p=1.0)),
        ("sunflare", RandomSunFlare(p=1.0)),
        ("coarsedropout", CoarseDropout(p=1.0, max_holes=10, max_height=32, max_width=32)),
        ("horizontal_flip", HorizontalFlip(p=1.0)),
        ("vertical_flip", VerticalFlip(p=1.0))
    ]

    for img_path, mask_path in tqdm(zip(images, masks), total=len(images), desc="Augmenting training data"):
        img_name = os.path.basename(img_path).split('.')[0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

        if img is not None and mask is not None:
            for aug_name, aug in augmentation_list:
                # Apply augmentation and preserve mask values
                augmented = Compose([aug], additional_targets={'mask': 'mask'})(image=img, mask=mask)
                aug_img, aug_mask = augmented['image'], augmented['mask']

                # Ensure mask values are preserved correctly after augmentation
                unique_before = np.unique(mask)
                unique_after = np.unique(aug_mask)
                print(f"Mask {img_name} - {aug_name}: Before {unique_before}, After {unique_after}")

                # Save augmented images and masks separately with original grayscale values
                cv2.imwrite(os.path.join(save_dir, 'images', f"{img_name}_{aug_name}.png"), aug_img)
                cv2.imwrite(os.path.join(save_dir, 'masks', f"{img_name}_{aug_name}.png"), aug_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Execute augmentation
dataset_path = '/path/to/dataset'
train_images, train_masks = load_training_data(dataset_path)

# Save augmented training data
augmented_dir = os.path.join(dataset_path, 'augmented_train')
os.makedirs(os.path.join(augmented_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(augmented_dir, 'masks'), exist_ok=True)

augment_training_data(train_images, train_masks, augmented_dir)
