import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import torch
from ultralytics import YOLO

# Seeding for reproducibility
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
height = 512
width = 768
num_classes = 3

# Paths
dataset_path = "/home/ahsan/University/Thesis/UNet_Directory/UNet_RoadSegmentation/final_thesis_code/data/test_dataset"
save_path = "/home/ahsan/University/Thesis/UNet_Directory/UNet_RoadSegmentation/final_thesis_code/data/combined_results"
yolo_model_path = "/home/ahsan/University/Thesis/UNet_Directory/UNet_RoadSegmentation/final_thesis_code/model_notebooks/best.pt"
unet_model_path = "/home/ahsan/University/Thesis/UNet_Directory/UNet_RoadSegmentation/final_thesis_code/model_notebooks/unet-multiclass.keras"

# Create Save Directory
os.makedirs(save_path, exist_ok=True)

# Load YOLO Model
yolo_model = YOLO(yolo_model_path)

# Load U-Net Model
unet_model = tf.keras.models.load_model(unet_model_path, compile=False)

# Load test images
test_x = sorted(glob(os.path.join(dataset_path, "images", "*.png")))

# Mapping grayscale values to class labels for segmentation mask
class_colors = {
    0: (0, 0, 0),       # Non-Drivable Area (black)
    1: (79, 247, 211),  # My Way (Greenish-Yellow)
    2: (247, 93, 79)    # Other Way (Red)
}

# Track inference times
time_taken = []

for x_path in tqdm(test_x, desc="Processing Images"):
    name = os.path.basename(x_path)
    original_img = cv2.imread(x_path, cv2.IMREAD_COLOR)

    if original_img is None:
        print(f"Error loading image {x_path}, skipping.")
        continue

    # Resize image for model input
    resized_img = cv2.resize(original_img, (width, height))
    img_input = resized_img / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    ####### 1. YOLO Object Detection #######
    yolo_results = yolo_model(x_path)[0]
    yolo_output = original_img.copy()

    for result in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())  # Bounding box coordinates
        confidence = result.conf[0].item()
        label = int(result.cls[0].item())

        # Draw bounding boxes
        cv2.rectangle(yolo_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(yolo_output, f"Class {label}: {confidence:.2f}", 
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save YOLO output image
    yolo_save_path = os.path.join(save_path, f"{name.split('.')[0]}_yolo.png")
    cv2.imwrite(yolo_save_path, yolo_output)

    ####### 2. U-Net Road Segmentation #######
    start_time = time.time()
    unet_prediction = unet_model.predict(img_input)[0]
    time_taken.append(time.time() - start_time)

    # Convert model output to mask
    predicted_mask = np.argmax(unet_prediction, axis=-1).astype(np.uint8)

    # Convert mask to color format
    seg_colored = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in class_colors.items():
        seg_colored[predicted_mask == label] = color

    # Save segmentation mask
    seg_save_path = os.path.join(save_path, f"{name.split('.')[0]}_segmentation.png")
    cv2.imwrite(seg_save_path, seg_colored)

    ####### 3. Combined Image (YOLO + Segmentation) #######
    combined_output = cv2.addWeighted(yolo_output, 0.6, seg_colored, 0.4, 0)

    # Save combined output
    combined_save_path = os.path.join(save_path, f"{name.split('.')[0]}_combined.png")
    cv2.imwrite(combined_save_path, combined_output)

# FPS Calculation
mean_time = np.mean(time_taken)
mean_fps = 1 / mean_time
print(f"Mean time taken per image: {mean_time:.4f} seconds")
print(f"Mean FPS: {mean_fps:.2f}")
