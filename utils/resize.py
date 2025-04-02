"""
This script resizes either images or segmentation labels using multiprocessing.

- It accepts four command-line arguments:
  1. mode: "image"/"images" for standard images or "label"/"labels" for segmentation labels.
  2. height: target height to resize to (width is adjusted to preserve aspect ratio).
  3. input_path: a file  or a folder.
  4. output_path: destination for the resized output, file or folder.

- For images: processes all files in a folder using bilinear interpolation.
- For labels: processes a single file or all files in a folder using nearest-neighbor interpolation to preserve class IDs.

The script uses all available CPU cores to speed up processing.
"""

from PIL import Image, ImageOps
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import sys

# Function to get image dimensions (width, height)
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

# Function to resize a single generic image
def resize_single_image(args):
    input_path, output_path, h = args
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        x, y = img.size
        target_size = (int(h * x / y), h)
        img_resized = img.resize(target_size, Image.BILINEAR)
        img_resized.save(output_path)

# Function to resize a single label image
def resize_single_label(args):
    input_path, output_path, h = args
    with Image.open(input_path) as img:
        x, y = img.size
        target_size = (int(h * x / y), h)
        img_resized = img.resize(target_size, Image.NEAREST)
        img_resized.save(output_path)

# Multiprocessing function for generic images
def resize_images(input_folder_path, output_folder_path, h):
    input_folder = Path(input_folder_path)
    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_files = list(input_folder.glob('*'))

    tasks = [(str(infile), str(output_folder / infile.name), h) for infile in input_files if infile.is_file()]

    with Pool(cpu_count()) as pool:
        pool.map(resize_single_image, tasks)

# Multiprocessing function for label images
def resize_labels(input_folder_path, output_folder_path, h):
    input_folder = Path(input_folder_path)
    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Recursively collect all label files in the folder
    input_files = [f for f in input_folder.rglob('*') if f.is_file()]

    tasks = [(str(infile), str(output_folder / infile.name), h) for infile in input_files]

    with Pool(cpu_count()) as pool:
        pool.map(resize_single_label, tasks)


# Command-line input handling using argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Resize images or labels.")
    parser.add_argument("mode", choices=["label", "labels", "image", "images"], help="Operation mode")
    parser.add_argument("height", type=int, help="Target height for resizing")
    parser.add_argument("input_path", type=str, help="Input file or folder path")
    parser.add_argument("output_path", type=str, help="Output file or folder path")

    args = parser.parse_args()

    mode = args.mode.lower()
    h = args.height
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if mode == "label":
        if not input_path.is_file():
            raise ValueError("Input path must be a file for 'label' mode.")
        resize_single_label((str(input_path), str(output_path), h))

    elif mode == "labels":
        if not input_path.is_dir():
            raise ValueError("Input path must be a directory for 'labels' mode.")
        resize_labels(str(input_path), str(output_path), h)

    elif mode == "image":
        if not input_path.is_file():
            raise ValueError("Input path must be a file for 'image' mode.")
        resize_single_image((str(input_path), str(output_path), h))

    elif mode == "images":
        if not input_path.is_dir():
            raise ValueError("Input path must be a directory for 'images' mode.")
        resize_images(str(input_path), str(output_path), h)

    else:
        raise ValueError("Unknown mode. Must be one of: label, labels, image, images, size.")
