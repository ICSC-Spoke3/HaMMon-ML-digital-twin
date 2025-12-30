# check_masks.py
# This script validates PNG segmentation masks using min and max allowed pixel values.
# It uses the utility functions from proc_masks.py.

import argparse
from pathlib import Path
import sys
import os

# Import proc_masks from the parent directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from src.proc_masks import check_folder

def validate_value(val_str):
    try:
        val = int(val_str)
        return val
    except ValueError:
        raise argparse.ArgumentTypeError("Value must be an integer.")

def main():
    parser = argparse.ArgumentParser(
        description="Validate PNG segmentation masks using min and max pixel values."
    )
    parser.add_argument("input_folder", type=Path, help="Path to the folder with PNG masks.")
    parser.add_argument("min_val", type=validate_value, help="Minimum allowed pixel value.")
    parser.add_argument("max_val", type=validate_value, help="Maximum allowed pixel value.")

    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Input folder '{args.input_folder}' is not a valid directory.")

    check_folder(args.input_folder, args.min_val, args.max_val)

    print("All masks validated successfully.")

if __name__ == "__main__":
    main()
