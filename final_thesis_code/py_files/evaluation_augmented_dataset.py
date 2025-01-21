import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score, accuracy_score

# Paths to predicted masks and ground truth masks
pred_mask_dir = "/path/to/predicted_masks"
true_mask_dir = "/path/to/true_masks"

# Load mask file paths
pred_mask_paths = sorted(glob(os.path.join(pred_mask_dir, "*.png")))
true_mask_paths = sorted(glob(os.path.join(true_mask_dir, "*.png")))

# Ensure equal number of files
assert len(pred_mask_paths) == len(true_mask_paths), "Mismatch in predicted and ground truth mask counts."

# Initialize metrics
scores = []

# Metrics Calculation
for pred_path, true_path in tqdm(zip(pred_mask_paths, true_mask_paths), total=len(pred_mask_paths), desc="Evaluating"):
    # Load predicted and true masks
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    true_mask = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)

    if pred_mask is None or true_mask is None:
        print(f"Skipping {pred_path} or {true_path} due to loading error.")
        continue

    # Flatten and ensure integer labels
    pred_mask = pred_mask.flatten().astype(np.int32)
    true_mask = true_mask.flatten().astype(np.int32)

    # Ensure mask values are within the expected range
    unique_pred = np.unique(pred_mask)
    unique_true = np.unique(true_mask)
    if not np.all(np.isin(unique_pred, [0, 1, 2])) or not np.all(np.isin(unique_true, [0, 1, 2])):
        print(f"Unexpected values in {pred_path}. Skipping...")
        continue

    # Calculate metrics
    acc = accuracy_score(true_mask, pred_mask)
    f1 = f1_score(true_mask, pred_mask, average="macro", labels=[0, 1, 2])
    iou = jaccard_score(true_mask, pred_mask, average="macro", labels=[0, 1, 2])
    recall = recall_score(true_mask, pred_mask, average="macro", labels=[0, 1, 2])
    precision = precision_score(true_mask, pred_mask, average="macro", labels=[0, 1, 2])

    # Append metrics
    scores.append([os.path.basename(pred_path), acc, f1, iou, recall, precision])

# Check if any valid scores were collected
if len(scores) == 0:
    print("No valid masks evaluated.")
    exit()

# Convert to numpy array for mean calculation
scores_np = np.array([s[1:] for s in scores])

# Calculate mean metrics
mean_scores = np.mean(scores_np, axis=0)

# Print Metrics
print(f"Mean Accuracy: {mean_scores[0]:0.5f}")
print(f"Mean F1 Score: {mean_scores[1]:0.5f}")
print(f"Mean IoU: {mean_scores[2]:0.5f}")
print(f"Mean Recall: {mean_scores[3]:0.5f}")
print(f"Mean Precision: {mean_scores[4]:0.5f}")

# Optional: Save detailed metrics to a file
output_file = "/path/to/evaluation_results.csv"
with open(output_file, "w") as f:
    f.write("Image,Accuracy,F1 Score,IoU,Recall,Precision\n")
    for score in scores:
        f.write(",".join(map(str, score)) + "\n")
    f.write(f"\nMean Metrics,,{mean_scores[0]:0.5f},{mean_scores[1]:0.5f},{mean_scores[2]:0.5f},{mean_scores[3]:0.5f},{mean_scores[4]:0.5f}\n")

print(f"Metrics saved to {output_file}")
