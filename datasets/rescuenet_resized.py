import os
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms

                #################
                ### RescueNet ###
                #################

DATASET_PATH = "/outputs/RescueNet"

class_names = ['Background', 'Water', 'Building_No_Damage', 'Building_Minor_Damage',
           'Building_Major_Damage', 'Building_Total_Destruction', 
           'Vehicle', 'Road-Clear', 'Road-Blocked', 'Tree','Pool']

class_colors = [
    (0, 0, 0),          # Background
    (61, 230, 250),      # Water
    (255, 0, 0),         # Building_No_Damage
    (180, 120, 120),     # Building_Minor_Damage
    (160, 150, 20),      # Building_Major_Damage
    (140, 140, 140),     # Building_Total_Destruction
    (255, 0, 245),       # Vehicle
    (255, 235, 0),       # Road-Clear
    (0, 255, 127),       # Road-Blocked (added a new distinguishable color)
    (0, 82, 255),        # Tree
    (4, 250, 7)          # Pool
]


# RGB STATS
mean = [134.0602882986386, 131.77490467005882, 121.64714905765645]
std = [67.86721142834165, 65.86607098636689, 65.6198659244834]

# NUMBER OF IMAGES EACH LABEL APPEARS IN:
image_count = [3406, 1254, 1523, 1365,  956,  729, 1354, 1988,  561, 1682,  263 ]
# NUMBER OF PIXELS FOR EACH LABEL:
pixel_count = [22679894325,  3509813882,  1143512180,  1136628428,   726309268,
          622711602,   143126533,  2992469634,   685492293,  9523743127,
           24513768 ]


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
                 split='train',
                 scale=713,
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

        self.path_dict = {
            "train": self.root_path+'semanticSegmentationTrainSet',
            "val": self.root_path+'semanticSegmentationValidationSet',
            "test": self.root_path+'semanticSegmentationTestSet'
        }

        self._add_to_dataset(self.path_dict[self.split])

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

