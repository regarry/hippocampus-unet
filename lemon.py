import cv2
import os
from pathlib import Path
import numpy as np

# Define the input and output folder paths
input_folder_imgs = './Data_hippo_02/imgs'      # Path to the input images folder
output_folder_imgs = './Data_hippo_02/imgs2'    # Path to the output images folder

input_folder_masks = './Data_hippo_02/masks'    # Path to the input masks folder
output_folder_masks = './Data_hippo_02/masks2'  # Path to the output masks folder

# Ensure the output folders exist
os.makedirs(output_folder_imgs, exist_ok=True)
os.makedirs(output_folder_masks, exist_ok=True)

# Set target size
target_size = (512, 512)

# Function to process RGB images into individual channel grayscale images and an averaged grayscale image
def process_rgb_image(input_path, output_folder, base_name):
    # Read the image
    image = cv2.imread(str(input_path))
    
    if image is None:
        print(f"Unable to read image file: {input_path}")
        return

    # Check if filename contains 'Nes'
    if 'Nes' in base_name:
        # Convert image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Save grayscale image with suffix _1
        output_path = os.path.join(output_folder, f"{base_name}_1.png")
        cv2.imwrite(output_path, grayscale_image)
        print(f"Converted '{input_path}' to grayscale and saved as '{output_path}'")
        
    else:
        # Process normally for RGB images
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Split into channels
        blue_channel, green_channel, red_channel = cv2.split(image)

        # Save each channel as a grayscale image
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_1.png"), red_channel)    # Red channel
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_2.png"), green_channel)  # Green channel
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_3.png"), blue_channel)   # Blue channel

        # Calculate and save the average grayscale image
        average_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_4.png"), average_gray)   # Averaged grayscale

        print(f"Processed RGB image: {input_path}")

# Function to process single-channel masks and resize them

def process_mask(input_path, output_path):
    # Read the mask image
    mask = cv2.imread(str(input_path))

    if mask is None:
        print(f"Unable to read mask file: {input_path}")
        return

    # Convert to single-channel grayscale if the mask has multiple channels
    if len(mask.shape) == 3 and mask.shape[2] == 3:  # RGB or multi-channel
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask[(mask > 0) & (mask < 255)] = 128
    mask = np.where(mask == 128, 128, np.where(mask >= 128, 255, 0))

    unique_values = sorted(set(mask.flatten()))
    print(f"Mask: {os.path.basename(input_path)} - Unique Grayscale Values: {unique_values}")

    # Resize the mask to the target size
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Save the processed mask
    cv2.imwrite(str(output_path), resized_mask)
    print(f"Processed and saved: {output_path}")

# Process all images in the imgs folder
for filename in os.listdir(input_folder_imgs):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder_imgs, filename)
        base_name = os.path.splitext(filename)[0]
        process_rgb_image(input_path, output_folder_imgs, base_name)

# Process all masks in the masks folder
for filename in os.listdir(input_folder_masks):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        input_path = os.path.join(input_folder_masks, filename)
        output_path = os.path.join(output_folder_masks, filename)
        process_mask(input_path, output_path)

print("All images and masks have been processed!")
