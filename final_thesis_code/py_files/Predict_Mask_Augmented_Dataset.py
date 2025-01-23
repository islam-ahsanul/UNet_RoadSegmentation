import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf

# Seeding for reproducibility
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
height = 512
width = 768
num_classes = 3

# Paths
dataset_path = "/path/to/test_dataset"
save_path = "/path/to/predicted_masks"
model_path = "/path/to/unet-multiclass.keras"

# Create Save Directory
os.makedirs(save_path, exist_ok=True)

# Load Model
model = tf.keras.models.load_model(model_path, compile=False)

# Load test images
test_x = sorted(glob(os.path.join(dataset_path, "images", "*.png")))

# Mapping grayscale values to class labels
class_colors = {
    0: 0,   # Non-Drivable Area (black)
    1: 128, # My Way (gray level 128)
    2: 255  # Other Way (white)
}

# Track inference times
time_taken = []

for x_path in tqdm(test_x, desc="Predicting Masks"):
    name = os.path.basename(x_path)
    x = cv2.imread(x_path, cv2.IMREAD_COLOR)

    if x is None:
        print(f"Error loading image {x_path}, skipping.")
        continue

    x = cv2.resize(x, (width, height))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict mask
    start_time = time.time()
    p = model.predict(x)[0]
    time_taken.append(time.time() - start_time)

    p = np.argmax(p, axis=-1).astype(np.uint8)

    print(f"Unique predicted values for {name}: {np.unique(p)}")  # Debugging

    # Create a grayscale mask
    p_gray = np.zeros((height, width), dtype=np.uint8)

    # Assign grayscale values based on class labels
    for label, gray_value in class_colors.items():
        p_gray[p == label] = gray_value

    # Save predicted mask in grayscale
    cv2.imwrite(os.path.join(save_path, f"{name.split('.')[0]}.png"), p_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# FPS Calculation
mean_time = np.mean(time_taken)
mean_fps = 1 / mean_time
print(f"Mean time taken per image: {mean_time:.4f} seconds")
print(f"Mean FPS: {mean_fps:.2f}")
