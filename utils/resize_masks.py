# preprocess_masks_resize.py
# This script validates and resizes segmentation masks in PNG format.
# It uses the utility functions from proc_imgs.py.

import argparse
from pathlib import Path
import sys
import os

# Import proc_imgs from the parent directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from src.proc_imgs import check_folder_image_sizes, resize_labels

def validate_size(size_str):
    try:
        size = int(size_str)
        if size <= 0:
            raise ValueError
        return size
    except ValueError:
        raise argparse.ArgumentTypeError("Size must be a positive integer.")

def main():
    parser = argparse.ArgumentParser(
        description="Validate and resize PNG segmentation masks to a target height."
    )
    parser.add_argument("input_folder", type=Path, help="Path to the input masks folder.")
    parser.add_argument("output_folder", type=Path, help="Path to save resized masks.")
    parser.add_argument("size", type=validate_size, help="Target height for resizing.")

    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Input folder '{args.input_folder}' is not a valid directory.")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    check_folder_image_sizes(args.input_folder)
    resize_labels(str(args.input_folder), str(args.output_folder), args.size)

    print(f"All masks resized to height {args.size} and saved in '{args.output_folder}'.")

if __name__ == "__main__":
    main()
