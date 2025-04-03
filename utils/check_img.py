# This script checks the dimensions of image files.
#
# - If a single image file is provided, it prints its dimensions.
# - If a folder is provided, it checks that all image files inside have the same dimensions.
#   If not, or if a non-image file is encountered, it raises an exception.
# - The folder check is parallelized using multiprocessing for performance.

from PIL import Image, UnidentifiedImageError
from pathlib import Path
import argparse
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
        print(f"All images have the same size: {sizes.pop()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check image size or consistency in a folder.")
    parser.add_argument("path", type=str, help="Path to an image file or a folder containing images")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        size = image_size(path)
        if size:
            print(f"Image size: {size[0]}x{size[1]} pixels")
    elif path.is_dir():
        check_folder_image_sizes(path)
    else:
        raise ValueError("The provided path is not valid.")
