import torch
import logging
from torch import nn
import os
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import dice_loss

from utils.dice_score import multiclass_dice_coeff, dice_coeff

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#torch.set_num_threads(4)
@torch.inference_mode()
def evaluate(net, dataloader, device, amp, alpha):
    net.eval()
    logging.debug('Starting evaluate.py')
    num_val_batches = len(dataloader)
    dice_score = 0
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    val_loss = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            logging.debug('Predicted mask')
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
                logging.debug('computed loss with criterion')
                loss += alpha * dice_loss(
                            F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                logging.debug('computed dice loss')
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                logging.debug('computed dice score')
                val_loss += loss.item()
                logging.debug('computed val_loss')

    net.train()
    return {'dice_score': (dice_score / max(num_val_batches, 1)),
            'val_loss': val_loss}
