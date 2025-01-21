import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf

# Seeding
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

# Load Model (update to .keras format)
model = tf.keras.models.load_model(model_path, compile=False)

# Define color map for visualization (BGR format)
colors = {
    0: (0, 0, 0),       # Non-Drivable Area (black)
    1: (79, 247, 211),  # My Way (#d3f74f) in BGR
    2: (247, 93, 79)    # Other Way (#4f5df7) in BGR
}

# Load test images
test_x = sorted(glob(os.path.join(dataset_path, "images", "*")))
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

    start_time = time.time()
    p = model.predict(x)[0]
    time_taken.append(time.time() - start_time)

    p = np.argmax(p, axis=-1).astype(np.uint8)  # Convert probabilities to class labels

    # Create an empty colored mask
    p_colored = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors based on class labels
    for label, color in colors.items():
        p_colored[p == label] = color

    # Save predicted mask
    cv2.imwrite(os.path.join(save_path, f"{name.split('.')[0]}.png"), p_colored)

# FPS Calculation
mean_time = np.mean(time_taken)
mean_fps = 1 / mean_time
print(f"Mean time taken: {mean_time:.4f} seconds")
print(f"Mean FPS: {mean_fps:.2f}")
