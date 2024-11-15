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
from utils.data_loading2 import BasicDataset
from utils.dice_score import dice_loss
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from matplotlib import pyplot as plt
from PIL import Image

#import wandb

#print(f'gpu count loc 0: {torch.cuda.device_count()}') 

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#gc.collect()
#torch.cuda.empty_cache()

# dir_rsc_storage = '/rsstu/users/t/tghashg/MADMbrains/Ryan/Pytorch-UNet/data'
# dir_img = Path(dir_rsc_storage+'/cfos_img_labelkit/')
# dir_mask = Path(dir_rsc_storage+'/cfos_mask_labelkit/')

dir_img = Path('./Data_hippo_02_1/train/imgs/')
dir_mask = Path('./Data_hippo_02_1/train/masks/')
dir_checkpoint = Path('./checkpoints3/')
dir_curves = Path('./learning_curves/')

def plot_loss(train_loss, val_loss, dice_score):
    epochs = np.array(range(1, len(train_loss) + 1))
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    # Convert each element in dice_score to CPU and then to NumPy array if it's a tensor
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
    
@torch.inference_mode()
def evaluate(net, dataloader, device, amp, alpha):
    net.eval()
    logging.info('Starting evaluate')
    num_val_batches = len(dataloader)
    dice_score = 0
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    val_loss = 0
    logging.info(f'criterion{criterion}')
    logging.info(f'device.type{device.type}')
    # iterate over the validation set
    #with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        logging.info('Got image and mask')
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        # predict the mask
        mask_pred = net(image)
        logging.info('Predicted mask')
        if net.n_classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            loss = criterion(mask_pred, mask_true)
            loss += alpha * dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            loss = criterion(mask_pred, mask_true)
            logging.info('computed loss with criterion')
            loss += alpha * dice_loss(
                        F.softmax(mask_pred, dim=1).float(),
                        F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
            logging.info('computed dice loss')
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            logging.info('computed dice score')
            val_loss += loss.item()
            logging.info('computed val_loss')
            
    logging.info('Evaluation finished')
    net.train()
    return {'dice_score': (dice_score / max(num_val_batches, 1)),
            'val_loss': val_loss}


def train_model(
        model,
        device,
        # experiment,
        epochs: int = 5,
        batch_size: int = 1,
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
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='', is_train=True)

    # 2. Split into train / validation partitions
    n_val = math.ceil(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, 
                                      [n_train, n_val], 
                                      generator=torch.Generator().manual_seed(16) )
    val_set.dataset.is_train = False
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)

    # experiment.config.update(
    #     dict(epochs=epochs, 
    #          batch_size=batch_size, 
    #          learning_rate=learning_rate,
    #          val_percent=val_percent, 
    #          save_checkpoint=save_checkpoint, 
    #          img_scale=img_scale, 
    #          amp=amp,
    #          alpha=alpha)
    # ) 
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, 
                              weight_decay=weight_decay, 
                              momentum=momentum, 
                              foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    train_loss_list = []
    val_loss_list = []
    dice_score_list = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
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
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                """
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_dict = evaluate(model, val_loader, device, amp, alpha)
                        #logging.info('Evaluation finished')
                        val_score = val_dict['dice_score']
                        val_loss = val_dict['val_loss']
                        #logging.info(f'val_loss: {val_loss}')
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))    
                """
            val_dict = evaluate(model, val_loader, device, amp, alpha)
            val_score = val_dict['dice_score']
            val_loss = val_dict['val_loss']
            logging.info(f'val_loss: {val_loss}')
            logging.info(f'val_score: {val_score}')
            scheduler.step(val_score)
            logging.info('Validation Dice score: {}'.format(val_score))  
            # wandb.log(
            #     {
            #         "train_loss": epoch_loss, 
            #         "val_loss": val_loss, 
            #         "dice_score": val_score
            #     }
            # )
                      
            train_loss_list.append(epoch_loss)
            val_loss_list.append(val_loss)
            dice_score_list.append(val_score)
            logging.info(f'''
                             epoch:      {epoch}
                             train loss: {epoch_loss}
                             val loss:   {val_loss}
                             dice score: {val_score}
                             ''')    
            
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            # wandb.save(str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
    plot_loss(train_loss_list,val_loss_list, dice_score_list)
    # wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=50, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--nchannels', type=int, default=1, help='Number of channels of input images (int)')

    return parser.parse_args()


if __name__ == '__main__':   
    args = get_args()
    
    logging.basicConfig(filename = 'logs/train.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    if not torch.cuda.is_available():
        sys.exit()

    model = UNet(n_channels=args.nchannels, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
        
        
    #print(f'gpu count loc 2: {torch.cuda.device_count()}')    
    #if torch.cuda.device_count() > 1:
        #i = list(range(0, torch.cuda.device_count()))
        #print(i)
        #torch.cuda.set_device(i)
        #model = nn.parallel.DistributedDataParallel(model,device_ids=[1]) # list of ints with the leading one requring the most memory for combining
    
    
    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            # experiment=experiment,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except KeyboardInterrupt:
        print('Hello user you have pressed ctrl-c button.')
        #plot_loss(train_loss_list,val_loss_list, dice_score_list)
    except Exception as e:
        print("Exception occurred: ", e)
        traceback.print_exc()
    """
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        gc.collect()
        torch.cuda.empty_cache()

        model.use_checkpointing()
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

    """