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
height = 768
width = 512
num_classes = 3

# Paths
dataset_path = "/path/to/test_dataset"
save_path = "/path/to/predicted_masks"
model_path = "/path/to/unet-multiclass.h5"

# Create Save Directory
os.makedirs(save_path, exist_ok=True)

# Load Model
model = tf.keras.models.load_model(model_path)

# Predict
test_x = sorted(glob(os.path.join(dataset_path, "images", "*")))
time_taken = []

for x_path in tqdm(test_x, desc="Predicting Masks"):
    name = os.path.basename(x_path)
    x = cv2.imread(x_path, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    start_time = time.time()
    p = model.predict(x)[0]
    time_taken.append(time.time() - start_time)

    p = np.argmax(p, axis=-1)  # Convert probabilities to class labels
    p_colored = cv2.applyColorMap((p * 85).astype(np.uint8), cv2.COLORMAP_JET)  # Visualize classes

    cv2.imwrite(os.path.join(save_path, name), p_colored)

# FPS Calculation
mean_time = np.mean(time_taken)
mean_fps = 1 / mean_time
print(f"Mean time taken: {mean_time} seconds")
print(f"Mean FPS: {mean_fps}")
