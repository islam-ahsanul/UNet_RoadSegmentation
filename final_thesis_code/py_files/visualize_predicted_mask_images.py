import cv2
import os
import matplotlib.pyplot as plt
from glob import glob

# Paths
original_image_dir = "/path/to/test_dataset/images"
pred_mask_dir = "/path/to/predicted_masks"
true_mask_dir = "/path/to/true_masks"

# Load test image paths
test_images = sorted(glob(os.path.join(original_image_dir, "*.png")))
predicted_masks = sorted(glob(os.path.join(pred_mask_dir, "*.png")))
true_masks = sorted(glob(os.path.join(true_mask_dir, "*.png")))

# Define grayscale color mapping
class_colors = {
    0: (0, 0, 0),       # Non-Drivable Area (black)
    128: (79, 247, 211),  # My Way (Greenish-Yellow)
    255: (247, 93, 79)    # Other Way (Red)
}

def visualize_sample(img_path, pred_mask_path, true_mask_path):
    # Read images
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert predicted grayscale values to RGB for visualization
    pred_colored = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    true_colored = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)

    for gray_value, color in class_colors.items():
        pred_colored[pred_mask == gray_value] = color
        true_colored[true_mask == gray_value] = color

    # Display original image, predicted mask, and true mask
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(pred_colored, cv2.COLOR_BGR2RGB))
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(true_colored, cv2.COLOR_BGR2RGB))
    plt.title("True Mask")
    plt.axis("off")

    plt.show()


# Visualize the first 5 images for comparison
for i in range(5):
    visualize_sample(test_images[i], predicted_masks[i], true_masks[i])
