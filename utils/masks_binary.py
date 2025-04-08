# create_binary_masks.py
# This script extracts binary masks for a specific label from PNG segmentation masks.
# It uses the utility functions from proc_masks.py.

import argparse
from pathlib import Path
import sys
import os

# Import proc_masks from the parent directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

sys.path.append(root_dir)

from src.proc_masks import process_folder

def validate_label(val_str):
    try:
        val = int(val_str)
        if val < 0 or val >= 255:
            raise ValueError
        return val
    except ValueError:
        raise argparse.ArgumentTypeError("Label must be an integer between 0 and 254.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract binary masks from PNG segmentation masks for a given label."
    )
    parser.add_argument("input_folder", type=Path, help="Path to the folder with PNG masks.")
    parser.add_argument("label_index", type=validate_label, help="Label index to extract (0-254).")
    parser.add_argument("output_name", type=str, help="Name for the output folder (without 'binary-' prefix).")

    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Input folder '{args.input_folder}' is not a valid directory.")

    output_dir = args.input_folder.parent / f"binary-{args.output_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    process_folder(args.input_folder, args.output_name, args.label_index)

if __name__ == "__main__":
    main()
