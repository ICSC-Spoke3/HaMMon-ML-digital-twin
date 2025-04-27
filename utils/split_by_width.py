# split_by_width.py

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import shutil

SOURCE_DIR = Path("/outputs/airflow_data/floodnet/img")

def get_image_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except UnidentifiedImageError:
        raise ValueError(f"Non-image file encountered: {path}")
    except Exception as e:
        raise RuntimeError(f"Error reading image {path}: {e}")

def split_images_by_width(source_folder: Path):
    if not source_folder.is_dir():
        raise ValueError("Provided path is not a directory")

    for image_path in source_folder.glob("*.*"):
        try:
            width, _ = get_image_size(image_path)
            target_dir = source_folder.parent / f"img-{width}"
            target_dir.mkdir(exist_ok=True)
            shutil.copy2(image_path, target_dir / image_path.name)
            print(f"Copied {image_path.name} to {target_dir}")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")

if __name__ == "__main__":
    split_images_by_width(SOURCE_DIR)
