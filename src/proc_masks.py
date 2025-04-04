"""
proc_masks.py

Utility functions for validating and manipulating PNG segmentation masks.

This module includes:

Validation utilities:
- load_and_check(image_path, min_val, max_val): Loads a .png mask and checks that it is valid (2D, within value range).
- check_folder(input_path, min_val, max_val): Validates all .png masks in a folder using multiprocessing.

Binary mask extraction:
- create_binary_mask(img_np, label, output_path): Creates a binary mask from a specific label in the input mask.
- process_folder(input_path, label_name, label_index): Processes all masks in a folder and extracts binary masks for a given label.

Internal helper functions (used for multiprocessing):
- _check_file(args): Wrapper for safely checking a single file.
- _process_file(args): Loads, extracts, and saves a binary mask from a single file.

All operations assume input masks are grayscale PNGs with integer labels.
"""


from PIL import Image
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

def load_and_check(image_path, min_val, max_val):
    """
    Load image and verify it is a valid segmentation mask:
    - Must be a .png file
    - Must exist and be a file
    - Must be 2D (single channel)
    - All pixel values must be within [min_val, max_val]

    Returns:
        np.ndarray

    Raises:
        ValueError if checks fail
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f"Path does not exist or is not a file: '{image_path}'")

    if image_path.suffix.lower() != ".png":
        raise ValueError(f"Only .png files are supported (got '{image_path.name}')")

    img = Image.open(image_path)
    img_np = np.array(img)

    if img_np.ndim != 2:
        raise ValueError(f"Image '{image_path}' is not 2D (ndim={img_np.ndim})")

    if img_np.min() < min_val or img_np.max() > max_val:
        raise ValueError(
            f"Pixel values out of range in '{image_path}': found [{img_np.min()}, {img_np.max()}], expected [{min_val}, {max_val}]"
        )

    return img_np

def create_binary_mask(img_np, label, output_path):
    """
    Create a binary mask from a given label:
    - Pixels equal to 'label' become 255
    - All others become 0

    Args:
        img_np (np.ndarray): Input segmentation mask (2D array)
        label (int): Label to extract
        output_path (Path): Path to save the output PNG

    Raises:
        ValueError: If output_path does not end with .png or already exists
    """
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output path must end with .png (got '{output_path}')")

    if output_path.exists():
        raise ValueError(f"Output file already exists: '{output_path}'")

    mask = np.where(img_np == label, 255, 0).astype(np.uint8)

    if np.any(mask):
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(output_path)
        return mask_img
    else:
        return None

def _check_file(args):
    image_path, min_val, max_val = args
    try:
        load_and_check(image_path, min_val=min_val, max_val=max_val)
        return None
    except Exception as e:
        return f"Error checking {image_path}: {e}"

def check_folder(input_path, min_val, max_val):
    """
    Run load_and_check on all files in the given folder using multiprocessing.

    Args:
        input_path (Path): Path to folder with images to check
        min_val (int): Minimum pixel value allowed
        max_val (int): Maximum pixel value allowed
    """
    input_path = Path(input_path)
    files = list(input_path.iterdir())
    args_list = [(f, min_val, max_val) for f in files]

    with Pool(cpu_count()) as pool:
        results = pool.map(_check_file, args_list)

    errors = [r for r in results if r]
    if errors:
        for e in errors:
            print(e)
        raise RuntimeError(f"{len(errors)} file(s) failed the check.")
    else:
        print("All files checked successfully.")

def _process_file(args):
    image_path, output_dir, label_index = args
    img_np = load_and_check(image_path, min_val=0, max_val=255)
    output_path = output_dir / (image_path.stem + "_mask.png")
    create_binary_mask(img_np, label_index, output_path)

def process_folder(input_path, label_name, label_index):
    """
    Process all images in input_path and create binary masks for a specific label.
    Creates a folder 'binary-LABELNAME' next to input_path.

    Args:
        input_path (Path): Path to folder with input masks
        label_name (str): Name of the label (used in output folder name)
        label_index (int): Index of the label to extract
    """
    input_path = Path(input_path)
    output_dir = input_path.parent / f"binary-{label_name}"
    output_dir.mkdir(exist_ok=False)

    files = list(input_path.iterdir())
    args_list = [(f, output_dir, label_index) for f in files]

    with Pool(cpu_count()) as pool:
        pool.map(_process_file, args_list)

    print(f"Binary masks saved to '{output_dir}'")
