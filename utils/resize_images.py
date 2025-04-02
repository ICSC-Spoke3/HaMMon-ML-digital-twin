from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm  # For progress bar

# Function to resize images
def resize_image(input_path, output_path, h):
    # Open an image file
    # with Image.open(input_path) as img:
    with default_loader(input_path) as img:
        img = ImageOps.exif_transpose(img) 
        x, y = img.size 
        target_size = (int(h*x/y), h)
        #print(f"size ({x},{y}) ---> {target_size}")
        # Resize the image
        img_resized = img.resize(target_size, Image.BILINEAR)  # Use NEAREST for segmentation masks (labels)
        # Save the resized image
        img_resized.save(output_path)
def resize_label(input_path, output_path, h):
    # Open an image file
    with Image.open(input_path) as img:
        x, y = img.size 
        target_size = (int(h*x/y), h)
        #print(f"size ({x},{y}) ---> {target_size}")
        # Resize the image
        img_resized = img.resize(target_size, Image.NEAREST)  # Use NEAREST for segmentation masks (labels)
        # Save the resized image
        img_resized.save(output_path)

target_height = 713
split = 'train'

for split in ['train','val','test']:
    # Directories for images and labels
    #                                                                           floodnet

    image_dir = Path(f'/datasets/FloodNet/FloodNet-Supervised_v1.0/{split}/{split}-org-img')
    label_dir = Path(f'/datasets/FloodNet/FloodNet-Supervised_v1.0/{split}/{split}-label-img')
    output_image_dir = Path(f'/outputs/FloodNet-resized-{target_height}/{split}/{split}-org-img')
    output_label_dir = Path(f'/outputs/FloodNet-resized-{target_height}/{split}/{split}-label-img')

    #                                                                           rescuenet
    # d = { "train":"semanticSegmentationTrainSet", "val":"semanticSegmentationValidationSet", "test":"semanticSegmentationTestSet"}
    # image_dir = Path(f'/datasets/RescueNet/{d[split]}/{split}-org-img')
    # label_dir = Path(f'/datasets/RescueNet/{d[split]}/{split}-label-img')
    # output_image_dir = Path(f'/outputs/RescueNet-resized-{target_height}/{d[split]}/{split}-org-img')
    # output_label_dir = Path(f'/outputs/RescueNet-resized-{target_height}/{d[split]}/{split}-label-img')

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(os.listdir(image_dir), desc=f"resizing {split} set"):
        if file_name[-4:] != '.jpg': raise ValueError('Wrong filename: ' + file_name)
        image_path = os.path.join(image_dir, file_name)
        label_path = os.path.join(label_dir, file_name[:-4]+'_lab.png')    
        output_image_path = os.path.join(output_image_dir, file_name)
        output_label_path = os.path.join(output_label_dir, file_name[:-4]+'_lab.png')
        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            raise ValueError(image_path + '\n' + label_path)
        resize_image(image_path, output_image_path, target_height)
        resize_label(label_path, output_label_path, target_height)
