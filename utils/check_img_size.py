from PIL import Image
from pathlib import Path
import sys

def print_image_size(image_path):
    path = Path(image_path)
    if not path.is_file():
        print("Error: the provided path is not a valid file.")
        return
    with Image.open(path) as img:
        width, height = img.size
        print(f"Image size: {width}x{height} pixels")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_size.py path/to/image.jpg")
    else:
        print_image_size(sys.argv[1])
