"""
proc_imgs.py

Utility functions for basic image processing tasks.

This module includes:

Functions for image dimension checks:
- image_size(image_path): Returns the (width, height) of a single image file.
- check_folder_image_sizes(folder_path): Verifies that all images in a folder have the same dimensions.

Functions for image resizing:
- resize_single_image(args): Resizes one standard image using bilinear interpolation.
- resize_single_label(args): Resizes one label/mask image using nearest-neighbor interpolation.
- resize_images(input_folder_path, output_folder_path, h): Batch-resizes standard images in a folder.
- resize_labels(input_folder_path, output_folder_path, h): Batch-resizes label images in a folder (recursive).

All resizing operations preserve the original aspect ratio.
Multiprocessing is used where applicable for performance.
"""


from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path
from multiprocessing import Pool, cpu_count

def image_size(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"The provided path is not a valid file: {image_path}")
    try:
        with Image.open(path) as img:
            return img.size
    except UnidentifiedImageError:
        raise ValueError(f"Non-image file encountered: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing file {image_path}: {e}")

def check_folder_image_sizes(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError("The provided path is not a valid directory.")

    image_files = list(folder.glob("*.*"))
    if not image_files:
        raise ValueError("No files found in the folder.")

    with Pool(cpu_count()) as pool:
        try:
            sizes = set(pool.map(image_size, image_files))
        except Exception as e:
            raise e

    if len(sizes) > 1:
        raise ValueError(f"Images have different sizes: {sizes}")
    else:
        size = sizes.pop()
        print(f"All images have the same size: {size}")
        return size

# Function to resize a single generic image
def resize_single_image(args):
    input_path, output_path, h = args
    try: 
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            x, y = img.size
            target_size = (int(h * x / y), h)
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_resized.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to resize image '{input_path}'") from e

# Function to resize a single label image
def resize_single_label(args):
    input_path, output_path, h = args
    try:
        with Image.open(input_path) as img:
            x, y = img.size
            target_size = (int(h * x / y), h)
            img_resized = img.resize(target_size, Image.NEAREST)
            img_resized.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to resize label image '{input_path}'") from e


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
