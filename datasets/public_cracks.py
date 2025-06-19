'/home/mauro/projects/HaMMon-EQ-private/data/public-datasets-splitted'

subfolders = [
    "CrackForest",
    "CrackLS315",
    "CrackTree260",
    "CRKWH100",
    "DeepCrack",
    "Stone331"
]

import os
from pathlib import Path
import yaml
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms
import logging

settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"

with settings_path.open('r') as f:
    settings = yaml.safe_load(f)
datasets_folder = settings.get('datasets_folder')

DATASET_PATH = datasets_folder + '/public-cracks'    

class_names = ['Background', 'Crack']

class_colors = [
    (0, 0, 0),          # Background
    (255, 255, 255),     # Crack
]

# RGB STATS
mean = [ 139.33827584656237, 138.5015996683192, 137.2962738262087]
std = [ 52.17905205505703, 51.580754955750486, 51.26065890449696]

# NUMBER OF IMAGES EACH LABEL APPEARS IN:
image_count = [1153, 1148]
# NUMBER OF PIXELS FOR EACH LABEL:
pixel_count = [297074300,   5177732]


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.from_numpy(np.array(pic, dtype=np.int64))
        return label



class Dataset(data.Dataset):
    root_path = DATASET_PATH
    
    # classes
    class_names = class_names
    class_colors = class_colors
    # stats
    mean = mean
    std = std
    image_count = image_count
    pixel_count = pixel_count

    def __init__(self, *,
                 split,
                 subfolders=subfolders,
                 img_transform=None,
                 joint_transform=None,
                 loader=default_loader):
        assert split in ('train', 'val', 'test')
        assert isinstance(subfolders, list) and len(subfolders) > 0, "subfolders must be a non-empty list"

        self.subfolders = subfolders
        self.split = split

        assert  os.path.exists(self.root_path), f'dataset not found {self.root_path}'

        self.loader = loader

        self.imgs = []

        self.norm_mean=[m/255.0 for m in self.__class__.mean]
        self.norm_std=[s/255.0 for s in self.__class__.std]

        self.transform = transforms.Compose([
            transforms.ToTensor(), # to [0,1] range
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        self.img_transform = img_transform 
        self.target_transform = LabelToLongTensor()
        self.joint_transform = joint_transform

        self.fulfill()
    
    def _get_path(self, index):
        path = self.imgs[index]
        target_path = path.replace('/imgs/', '/labels/')
        target_path = target_path.replace('.jpg', '.png')
        return path, target_path

    def __getitem__(self, index):
        path, target_path = self._get_path(index) 
        img = self.loader(path)
        img = ImageOps.exif_transpose(img) 
        target = Image.open(target_path)


        if self.joint_transform is not None: 
            img, target = self.joint_transform([img, target])

        if self.img_transform is not None:
            img = self.img_transform(img)

        img = self.transform(img)
        target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.imgs)


    def fulfill(self):
        """
        Fills the dataset with images and their corresponding labels.
        """
        for subfolder in self.subfolders:
            dir = os.path.join(self.root_path, subfolder, self.split, 'imgs')
            assert os.path.exists(dir), f"Subfolder {subfolder} not found in {dir}"

            for root, _, fnames in sorted(os.walk(dir)):
                for fname in fnames:
                    if is_image_file(fname):
                        # # purge the .jpg extension from the label path
                        # fname = fname[:-4] 
                        path = os.path.join(root, fname)
                        self.imgs.append(path)

        logging.info(f"Dataset loaded with {len(self.imgs)} images.")
