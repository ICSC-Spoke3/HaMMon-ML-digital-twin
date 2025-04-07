# Resize Stage Script
# This script validates and resizes all images in a folder to a target height,
# preserving aspect ratio. It is designed as the first step in a computer vision pipeline.
#
# Input:
#   - input_folder: a directory containing only image files, all with identical dimensions.
#   - output_folder: destination directory where resized images will be saved.
#   - size: integer height (in pixels) to which all images will be resized, preserving aspect ratio.
#
# Notes:
#   - The script will fail if the input folder contains non-image files or images with inconsistent sizes.
#   - The output folder will be created if it does not exist.

import argparse
from pathlib import Path

import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(root_dir)

from src.proc_imgs import check_folder_image_sizes, resize_images

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
        description="Resize images in a folder to a given height while preserving aspect ratio."
    )
    parser.add_argument("input_folder", type=Path, help="Path to the input image folder.")
    parser.add_argument("output_folder", type=Path, help="Path to save resized images.")
    parser.add_argument("size", type=validate_size, help="Target height for resizing.")

    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Input folder '{args.input_folder}' is not a valid directory.")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    # check_folder_image_sizes and resize_images raise appropriate exceptions
    check_folder_image_sizes(args.input_folder)
    resize_images(str(args.input_folder), str(args.output_folder), args.size)

    print(f"Images resized successfully to height {args.size} and saved in '{args.output_folder}'.")

if __name__ == "__main__":
    main()
