import logging
import numpy as np
import torch
import random
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import re

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', is_train: bool = True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.is_train = is_train
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Get list of image files and group by base name
        self.image_groups = {}
        for file in listdir(images_dir):
            if isfile(join(images_dir, file)) and not file.startswith('.'):
                base_name = re.match(r'(.+)_\d+$', splitext(file)[0]).group(1)
                if base_name in self.image_groups:
                    self.image_groups[base_name].append(file)
                else:
                    self.image_groups[base_name] = [file]
        
        if not self.image_groups:
            raise RuntimeError(f'No input files found in {images_dir}. Please make sure your images are there.')
        
        self.mask_values = self.get_mask_values(mask_dir)
        
    def get_mask_values(self, mask_dir):
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=mask_dir, mask_suffix=self.mask_suffix), self.image_groups.keys()),
                total=len(self.image_groups)
            ))
        flattened_unique = [arr.flatten() for arr in unique]
        mask_value = list(sorted(np.unique(np.concatenate(flattened_unique)).tolist()))
        return mask_value


    # def __len__(self):
    #     return len(self.ids)
    def __len__(self):
        return sum(len(images) for images in self.image_groups.values())

    def random_rotation_augment(self, img, mask, p_flip = 0.5):
        """ rotate numpy array pairs randomly in 90 degree increments"""
        assert img.shape == mask.shape, f"image and mask should have the same shape instead of image size: {img.shape} and mask size: {mask.shape}"
        # Generate a random number to decide the angle of rotation
        upper_limit = int(1/(p_flip+1/10000) * 4)
        num_rotations = np.random.randint(1, upper_limit)
        
        # Rotate the image and mask
        img_rotated = np.rot90(img, num_rotations)
        mask_rotated = np.rot90(mask, num_rotations)

        return img_rotated, mask_rotated
    
    @staticmethod
    def random_gradient_augment(img, p_gradient=0.5, min_gradient = 0.2):
        """ randomly augment a grayscale image numpy array with a random gradient. 
        Gradients can be linear horizontal, linear vertical, or radial. """
        
        # Generate a random number to decide the type of gradient'
        upper_limit = int(1/(p_gradient+1/10000) * 6)
        gradient_type = np.random.randint(0, upper_limit)
        
        # Create the gradient
        if gradient_type == 0:  # linear horizontal +
            gradient = np.linspace(min_gradient, 1, img.shape[1], endpoint=True)
            gradient = np.tile(gradient, (img.shape[0], 1))
        elif gradient_type == 1:  # linear vertical +
            gradient = np.linspace(min_gradient, 1, img.shape[0], endpoint=True)
            gradient = np.tile(gradient, (img.shape[1], 1)).T
        elif gradient_type == 2:  # radial +
            y_indices, x_indices = np.indices(img.shape)
            center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
            gradient = np.hypot(x_indices - center_x, y_indices - center_y)
            gradient = gradient / gradient.max()
            gradient = np.maximum(gradient, min_gradient)
        elif gradient_type == 3:  # linear horizontal -
            gradient = np.linspace(1, min_gradient, img.shape[1], endpoint=True)
            gradient = np.tile(gradient, (img.shape[0], 1))
        elif gradient_type == 4:  # linear vertical -
            gradient = np.linspace(1, min_gradient, img.shape[0], endpoint=True)
            gradient = np.tile(gradient, (img.shape[1], 1)).T
        elif gradient_type == 5:  # radial -
            y_indices, x_indices = np.indices(img.shape)
            center_x, center_y = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
            gradient = np.hypot(x_indices - center_x, y_indices - center_y)
            gradient = (gradient.max() - gradient)/gradient.max()
            gradient = np.maximum(gradient, min_gradient)
        else: # no gradient
            gradient = np.ones(img.shape)
            
        # Apply the gradient to the image
        aug_img = img * gradient
        # Convert the gradient to the same data type as the img array
        aug_img = aug_img.astype(np.uint8)

        return aug_img
    
    @staticmethod
    def random_flip_augment(img, mask):
        """Takes tensor of image and mask and augments
        the image with random pytorch transforms"""
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(360),
        ])
        state = torch.get_rng_state()  
        img = transform(img)
        torch.set_rng_state(state)
        mask = transform(mask)
        return img, mask
    
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, is_train=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        arr = np.asarray(pil_img)
            
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if arr.ndim == 2:
                    mask[arr == v] = i
                else:
                    mask[(arr == v).all(-1)] = i

            return mask

        else: # this is if the data is an image not mask
            if is_train:
                # only augment images for training dataset
                arr = BasicDataset.random_gradient_augment(arr, p_gradient=0.5, min_gradient = 0.2)
                
            # if arr.ndim == 2:
            #     img = arr[np.newaxis, ...]
            # else:
            #     img = arr.transpose((2, 0, 1))

            img=arr
            if (img > 1).any():
                img = img / 255.0

            return img


    def __getitem__(self, idx):

        base_name = list(self.image_groups.keys())[idx // len(self.image_groups)]
        image_name = self.image_groups[base_name][idx % len(self.image_groups[base_name])]

        # name = self.ids[idx]
        # mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))

        img_file = self.images_dir / image_name
        mask_file = self.mask_dir / f"{base_name}{self.mask_suffix}.png"

        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {image_name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {base_name}: {mask_file}'
        
        mask = load_image(mask_file)
        img = load_image(img_file)
        
        assert img.size == mask.size, \
            f'Image and mask {image_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, is_train=self.is_train)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, is_train=self.is_train)
            
        if img.ndim == 2:
            img = img[np.newaxis, ...]  

        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        #打印通道信息
        # print(f"Image {name} loaded with shape: {img.shape}")
        # print(f"Mask {name} loaded with shape: {mask.shape}")
        
        mean, std = img.mean([1,2]), img.std([1,2])

        if torch.any(std == 0):
            logging.warning(f"Standard deviation is zero for image {image_name}.") 

        transform_norm = transforms.Normalize(mean, std)
        img = transform_norm(img)
        
        if self.is_train: # only augment images for training dataset
            img, mask = self.random_flip_augment(img, mask)
        
        return {
            'image': img,
            'mask': mask
        }
