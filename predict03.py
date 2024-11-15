import argparse 
import logging
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.data_loading03 import BasicDataset
from unet import UNet
from utils.dice_score import dice_coeff
import csv
from utils.dice_score import multiclass_dice_coeff  

# Target size for inference
TARGET_SIZE = (256, 256)

def visualize_with_dice(original_img, mask_pred, mask_gt, dice_score, filename, output_dir):
    # Resize masks to original image size for visualization
    mask_pred_resized = Image.fromarray(mask_pred.astype(np.uint8)).resize(original_img.size, Image.NEAREST)
    mask_gt_resized = Image.fromarray(mask_gt).resize(original_img.size, Image.NEAREST)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(mask_pred_resized, cmap='gray')
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")
    
    axs[2].imshow(mask_gt_resized, cmap='gray')
    axs[2].set_title("Ground Truth Mask")
    axs[2].axis("off")
    
    fig.suptitle(f"Dice Score: {dice_score:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_viz.png")
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

def predict_img(net, full_img, device, out_threshold=0.5):
    # Save original size for resizing output masks later
    original_size = full_img.size
    # Resize input image to TARGET_SIZE for model inference
    full_img_resized = full_img.resize(TARGET_SIZE, Image.BILINEAR)
    print(f"Input image resized to: {full_img_resized.size} for model inference")

    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img_resized, scale=1, is_mask=False))
    
    if img.dim() == 2:  
        img = img.unsqueeze(0)  
    elif img.dim() == 3 and img.shape[0] == 1:  
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    mean, std = img.mean([1,2]), img.std([1,2])
    transform_norm = transforms.Normalize(mean, std)
    img = transform_norm(img)
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    # Return resized mask to original size and the processed mask at TARGET_SIZE
    return mask[0].long().squeeze().numpy(), original_size

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')#nargs='+', 
    parser.add_argument('--ground_truth', '-g', metavar='GROUND_TRUTH', help='Directory of ground truth masks', required=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    
    return parser.parse_args()


def get_png_files(directory):
    return glob.glob(os.path.join(directory, '*.png'))

def get_png_files(directory):
    # Use glob to match .png file paths
    file_paths = glob.glob(os.path.join(directory, '*.png'))
    return file_paths

if __name__ == '__main__':

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = get_png_files(args.input[0])

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device,weights_only=True)
    mask_values = state_dict.pop('mask_values', [0, 1, 2])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    results = []
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    
    for filename in in_files:
        logging.info(f'Processing image {filename} ...')
        img = Image.open(filename)
        mask_pred, original_size = predict_img(net=net, full_img=img, out_threshold=args.mask_threshold, device=device)

        base_filename = os.path.basename(filename)
        gt_filename = os.path.join(args.ground_truth, base_filename.rsplit('_', 1)[0] + ".png")
        
        if not os.path.exists(gt_filename):
            logging.warning(f'Ground truth file {gt_filename} does not exist. Skipping...')
            continue

        mask_gt = Image.open(gt_filename).resize(TARGET_SIZE, Image.NEAREST)
        mask_gt = np.array(mask_gt)
        mask_gt[mask_gt == 128] = 1
        mask_gt[mask_gt == 255] = 2    

        mask_pred_tensor = torch.tensor(mask_pred).long()
        mask_gt_tensor = torch.tensor(mask_gt).long()

        if net.n_classes == 1:
            dice_score = dice_coeff(mask_pred_tensor, mask_gt_tensor, reduce_batch_first=False)
        else:
            mask_pred_one_hot = F.one_hot(mask_pred_tensor, net.n_classes).permute(2, 0, 1).unsqueeze(0).float()
            mask_gt_one_hot = F.one_hot(mask_gt_tensor, net.n_classes).permute(2, 0, 1).unsqueeze(0).float()
            dice_score = multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_gt_one_hot[:, 1:], reduce_batch_first=False)

        results.append((base_filename, dice_score.item()))

        if args.viz:
            visualize_with_dice(img, mask_pred, np.array(mask_gt), dice_score, filename,output_dir=args.output)

    csv_filename = "dice_scores.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Dice Score"])
        writer.writerows(results)

    logging.info(f'Dice scores saved to {csv_filename}')