import shutil
import random
from pathlib import Path
from PIL import Image
##只有训练集改变图像大小，验证集不变
# 定义图像和掩码的源文件夹
source_dir_imgs = Path('./Data_hippo_02/imgs')
source_dir_masks = Path('./Data_hippo_02/masks')

# 定义输出文件夹用于存放划分后的数据集
output_dir = Path('./Data_hippo_02_2')
train_dir_imgs = output_dir / 'train' / 'imgs'
train_dir_masks = output_dir / 'train' / 'masks'
val_dir_imgs = output_dir / 'val' / 'imgs'
val_dir_masks = output_dir / 'val' / 'masks'

# 创建输出文件夹（如果不存在）
train_dir_imgs.mkdir(parents=True, exist_ok=True)
train_dir_masks.mkdir(parents=True, exist_ok=True)
val_dir_imgs.mkdir(parents=True, exist_ok=True)
val_dir_masks.mkdir(parents=True, exist_ok=True)

# 收集所有图像文件
image_files = list(source_dir_imgs.glob('*.png'))

# 随机选择70%的图像用于训练集
train_files = random.sample(image_files, int(len(image_files) * 0.7))

# 剩下的图像用于验证集
val_files = [img for img in image_files if img not in train_files]

# 处理图像和掩码文件：将它们转换为灰度图像，并根据是否为训练集调整大小
def process_and_copy_image_and_mask(image_file, target_img_dir, target_mask_dir, resize=False):
    # 打开图像并转换为灰度
    img = Image.open(image_file).convert('L')
    if resize:
        img = img.resize((256, 256), Image.BICUBIC)
    img.save(target_img_dir / image_file.name)

    # 查找对应的掩码文件
    mask_file = source_dir_masks / f"{image_file.stem}.png"
    if mask_file.exists():
        # 打开掩码图像并转换为灰度
        mask = Image.open(mask_file).convert('L')
        if resize:
            mask = mask.resize((256, 256), Image.NEAREST)
        
        # 将掩码灰度值统一为 [0, 128, 255]
        mask_arr = mask.load()  # 获取像素访问权限
        width, height = mask.size
        for x in range(width):
            for y in range(height):
                if 0 < mask_arr[x, y] < 255:
                    mask_arr[x, y] = 128  # 将灰度值范围0-255之间的设为128

        # 保存处理后的掩码
        mask.save(target_mask_dir / mask_file.name)
    else:
        print(f"未找到对应掩码文件: {mask_file.name} 对于图像 {image_file.name}")

# 处理并复制训练集文件（70%），将图像和掩码调整为256x256
for image_file in train_files:
    process_and_copy_image_and_mask(image_file, train_dir_imgs, train_dir_masks, resize=True)

# 处理并复制验证集文件（30%），保持原始尺寸
for image_file in val_files:
    process_and_copy_image_and_mask(image_file, val_dir_imgs, val_dir_masks, resize=False)

print("数据集划分和处理成功！")
