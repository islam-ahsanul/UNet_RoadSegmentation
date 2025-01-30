import os
from glob import glob
import cv2

def process_and_rename_images(input_folder, output_folder, start_number=118):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files (supporting common formats)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob(os.path.join(input_folder, ext)))
        image_files.extend(glob(os.path.join(input_folder, ext.upper())))
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    # Process each image
    current_number = start_number
    for img_path in image_files:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Resize image
        img_resized = cv2.resize(img, (768, 512))
        
        # Create new filename with 4 digits
        new_filename = f"{current_number:04d}.png"
        output_path = os.path.join(output_folder, new_filename)
        
        # Save image as PNG
        cv2.imwrite(output_path, img_resized)
        print(f"Processed: {img_path} -> {new_filename}")
        
        current_number += 1
    
    print(f"Processed {current_number - start_number} images")
    print(f"Last image number: {current_number - 1}")

# Example usage
input_folder = "/home/ahsan/University/Thesis/UNet_Directory/Datasets/queue"
output_folder = "/home/ahsan/University/Thesis/UNet_Directory/Datasets/cleaned_images"
process_and_rename_images(input_folder, output_folder, start_number=118)