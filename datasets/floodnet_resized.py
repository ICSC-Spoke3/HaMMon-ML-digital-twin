import os
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms

                ################
                ### FloodNet ###
                ################

DATASET_PATH = "/outputs/FloodNet"

class_names = ['Background', 'Building-flooded', 'Building-not-flooded', 'Road-flooded',
           'Road-not-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']
class_colors = [
    (0, 0, 0),          # background
    (255, 0, 0),        # building_flooded
    (180, 120, 120),    # building_non_flooded
    (160, 150, 20),     # road_flooded
    (140, 140, 140),    # road_non_flooded
    (61, 230, 250),     # water
    (0, 82, 255),       # tree
    (255, 0, 245),      # vehicle
    (255, 235, 0),      # pool
    (4, 250, 7)         # grass
]

# RGB STATS
# Mean RGB values before preprocessing: R: 104.60373793316728, G: 114.01185437928378, B: 87.07508388806389
# Standard Deviation RGB values before preprocessing: R: 53.16452036870261, G: 49.371872909878796, B: 53.341147332404525
mean = [104.60373793316728, 114.01185437928378, 87.07508388806389]
std = [53.16452036870261, 49.371872909878796, 53.341147332404525]

# NUMBER OF IMAGES EACH LABEL APPEARS IN:
image_count = [  98,  149,  540,  162,  711,  668, 1156,  496,  331, 1331]
# NUMBER OF PIXELS FOR EACH LABEL:
pixel_count = [ 308842999,  318505750,  572544673,  559209008,  966381628, 1979142780,
        3107988573,   32624508,   36997059, 9914900430]


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
                 scale=713,
                 split='train',
                 joint_transform=None,
                 loader=default_loader):
        assert split in ('train', 'val', 'test')

        self.split = split
        
        self.root_path = self.root_path + f"-resized-{scale}/"
        assert  os.path.exists(self.root_path), f'dataset not found {self.root_path}'

        self.loader = loader

        self.imgs = []

        self.norm_mean=[m/255.0 for m in self.__class__.mean]
        self.norm_std=[s/255.0 for s in self.__class__.std]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        self.target_transform = LabelToLongTensor()
        self.joint_transform = joint_transform


        self._add_to_dataset(os.path.join(self.root_path, self.split))

    def _get_path(self, index):
        path = self.imgs[index]
        target_path = path.replace('-org-img/', '-label-img/')
        target_path = target_path.replace('.jpg', '_lab.png')
        return path, target_path
    

    def __getitem__(self, index):
        path, target_path = self._get_path(index)
        img = self.loader(path)
        img = ImageOps.exif_transpose(img) 
        target = Image.open(target_path)

        if self.joint_transform is not None: 
            img, target = self.joint_transform([img, target])

        img = self.transform(img)
        target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def _add_to_dataset(self, dir):
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if '-org-img' in root and is_image_file(fname):
                    path = os.path.join(root, fname)
                    self.imgs.append(path)

