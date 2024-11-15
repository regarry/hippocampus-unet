import os
import shutil
import random
from pathlib import Path

import os
import shutil
import random
from pathlib import Path

# Define source directories for images and masks
source_dir_imgs = Path('./Data_hippo_02/imgs2')
source_dir_masks = Path('./Data_hippo_02/masks2')

# Define output directories for the split dataset
output_dir = Path('./Data_hippo_02_1')
train_dir_imgs = output_dir / 'train' / 'imgs'
train_dir_masks = output_dir / 'train' / 'masks'
val_dir_imgs = output_dir / 'val' / 'imgs'
val_dir_masks = output_dir / 'val' / 'masks'

# Ensure the output directories exist
train_dir_imgs.mkdir(parents=True, exist_ok=True)
train_dir_masks.mkdir(parents=True, exist_ok=True)
val_dir_imgs.mkdir(parents=True, exist_ok=True)
val_dir_masks.mkdir(parents=True, exist_ok=True)

# Collect all image files in imgs2
image_files = list(source_dir_imgs.glob('*.png'))

# Randomly select 70% of the images for training
train_files = random.sample(image_files, int(len(image_files) * 0.7))

# Use the remaining images for validation
val_files = [img for img in image_files if img not in train_files]

# Function to copy an image and its corresponding mask
def copy_image_and_mask(image_file, target_img_dir, target_mask_dir):
    # Copy the image to the target directory
    shutil.copy(image_file, target_img_dir / image_file.name)
    
    # Find and copy the corresponding mask based on the base name
    base_name = '_'.join(image_file.stem.split('_')[:-1])
    mask_file = source_dir_masks / f"{base_name}.png"
    if mask_file.exists():
        shutil.copy(mask_file, target_mask_dir / mask_file.name)
    else:
        print(f"Mask file {mask_file.name} not found for image {image_file.name}")

# Copy files for the training set (70%)
for image_file in train_files:
    copy_image_and_mask(image_file, train_dir_imgs, train_dir_masks)

# Copy files for the validation set (30%)
for image_file in val_files:
    copy_image_and_mask(image_file, val_dir_imgs, val_dir_masks)

print("Dataset split and copy completed successfully!")

