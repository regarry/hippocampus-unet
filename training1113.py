import argparse
import logging
import os
import random
import sys
import gc
import traceback
import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet import UNet
from utils.data_loading04 import BasicDataset
from utils.dice_score import dice_loss
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from matplotlib import pyplot as plt
from PIL import Image

# Setting paths
dir_img = Path('./Data_hippo_02_4/train/imgs/')
dir_mask = Path('./Data_hippo_02_4/train/masks/')
dir_checkpoint = Path('./checkpoints14/')
dir_curves = Path('./learning_curves/')

# Configure logging to output to train.log
time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"train_{time}.log"

logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Logging initialized with time-stamped filename.")

# Define plotting function
def plot_loss(train_loss, val_loss, dice_score):
    epochs = np.array(range(1, len(train_loss) + 1))
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    if isinstance(dice_score, list):
        dice_score = np.array([score.cpu().numpy() if torch.is_tensor(score) else score for score in dice_score])
    else:
        dice_score = np.array(dice_score.cpu().numpy() if torch.is_tensor(dice_score) else dice_score)

    plt.plot(epochs, train_loss / train_loss[0], label='train loss')
    plt.plot(epochs, val_loss / val_loss[0], label='val loss')
    plt.plot(epochs, dice_score, label='dice score')
    plt.title("Normalized Loss Plot vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss: Cross Entropy + Dice")
    plt.legend()
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(dir_curves).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{dir_curves}/learning_curve_{current_datetime}.png')
    plt.show() 

# Evaluation function
@torch.inference_mode()
def evaluate(net, dataloader, device, amp, alpha):
    net.eval()
    logging.info('Starting evaluate')
    num_val_batches = len(dataloader)
    dice_score = 0
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    val_loss = 0
    logging.info(f'criterion{criterion}')
    #logging.info(f'device.type{device.type}')

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        logging.info('Got image and mask')
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        mask_pred = net(image)
        logging.info('Predicted mask')
        if net.n_classes == 1:
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            loss = criterion(mask_pred, mask_true)
            loss += alpha * dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            loss = criterion(mask_pred, mask_true)
            loss += alpha * dice_loss(
                        F.softmax(mask_pred, dim=1).float(),
                        F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            val_loss += loss.item()
    logging.info(f'Epoch Validation Dice Score: {dice_score / max(num_val_batches, 1):.4f}, Validation Loss: {val_loss:.4f}')
    logging.info('Evaluation finished')
    net.train()
    return {'dice_score': (dice_score / max(num_val_batches, 1)),
            'val_loss': val_loss}

# Main training function
def train_model(
        model,
        device,
        epochs: int = 100,
        batch_size: int = 50,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        alpha: float = 1
):
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='', is_train=True)
    n_val = math.ceil(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(16))
    val_set.dataset.is_train = False

    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    class_weights = torch.tensor([1.0, 2.0, 5.0], device=device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    train_loss_list = []
    val_loss_list = []
    dice_score_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                assert true_masks.min() >= 0 and true_masks.max() < model.n_classes, \
                f"Expected true_masks values in range [0, {model.n_classes - 1}], found [{true_masks.min()}, {true_masks.max()}]"    

                masks_pred = model(images)
                if model.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += alpha * dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += alpha * dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            val_dict = evaluate(model, val_loader, device, amp, alpha)
            val_score = val_dict['dice_score']
            val_loss = val_dict['val_loss']
            scheduler.step(val_score)

            train_loss_list.append(epoch_loss)
            val_loss_list.append(val_loss)
            dice_score_list.append(val_score)

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
    
    logging.info(f'Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Score: {val_score:.4f}')
    plot_loss(train_loss_list, val_loss_list, dice_score_list)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=50, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--nchannels', type=int, default=1, help='Number of channels of input images (int)')
    return parser.parse_args()

if __name__ == '__main__':   
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to CPU
    #logging.info(f'Using device {device}')

    model = UNet(n_channels=args.nchannels, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device=device)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
        
    except KeyboardInterrupt:
        print('Training interrupted by user.')

    except Exception as e:
        print("Exception occurred: ", e)
        traceback.print_exc()
