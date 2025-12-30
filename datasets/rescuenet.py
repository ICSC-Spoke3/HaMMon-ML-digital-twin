import os
from pathlib import Path
import yaml
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms


                #################
                ### RescueNet ###
                #################
"""
RescueNet is a high-resolution UAV image dataset captured after Hurricane Michael (October 2018), 
tailored for post-disaster semantic segmentation. 
It contains 4,494 UAV images (3,595 train / 449 val / 450 test), 
each with pixel-level labels across 10 classes.

GitHub repository: https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation
arXiv paper: https://arxiv.org/abs/2202.12361
Scientific Data (Nature): https://www.nature.com/articles/s41597-023-02743-2

# download link (June 2025)
curl -L -o ./rescuenet.zip https://www.kaggle.com/api/v1/datasets/download/yaroslavchyrko/rescuenet

Expected layout:

semanticSegmentationTrainSet/
.../-org-img/*.jpg
.../-label-img/*_lab.png
semanticSegmentationValidationSet/ (same structure)
semanticSegmentationTestSet/ (same structure)

If you want to use smaller resolutions, resize both images and masks ahead of time,
place the results under RescueNet-resized-{SCALE} at the dataset root, and keep the standard subfolders

"""


settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"

if settings_path.exists():
    with settings_path.open('r') as f:
        settings = yaml.safe_load(f)
    datasets_folder = Path(settings.get('datasets_folder'))
else: 
    datasets_folder = None


DATASET_PATH = f"{datasets_folder}/RescueNet"

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
 

    @classmethod
    def stats_from_yaml(cls, path):

        if path is None:
            raise ValueError("Path to stats file must be defined. Please set the path in settings.yaml.")
        else:
            path = Path(path)
            if not path.is_absolute():
                path = Path(__file__).parent / 'stats' / path

        assert path.exists(), f"Stats file {path} not found."
        with path.open('r') as f:
            stats = yaml.safe_load(f)

        # verify the required keys are present
        required_keys = ['mean', 'std']
        for key in required_keys:
            assert key in stats, f"Key '{key}' not found in stats file {path}." 
        
        cls.mean = stats['mean']
        cls.std = stats['std']
        cls.norm_mean = [m / 255.0 for m in cls.mean]
        cls.norm_std = [s / 255.0 for s in cls.std]
        cls.image_count = stats.get('image_count', None)
        cls.pixel_count = stats.get('pixel_count', None)

    @classmethod
    def transform(cls):
        """
        Image transform: converts to tensor and normalizes using dataset RGB stats.
        """

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.norm_mean, std=cls.norm_std)
        ])

    @classmethod
    def load_img(cls, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert('RGB')

    @classmethod
    def load_target(cls, path):
        target = Image.open(path)
        target = ImageOps.exif_transpose(target)
        return target.convert('L')
    

    def __init__(self, *,
                 split,
                 scale,
                 img_transform=None,
                 joint_transform=None,
                 loader=default_loader):
        assert split in ('train', 'val', 'test')

        self.split = split

        # if scale is nor 3000 nor none, use resized folder
        if scale not in (None, 3000):
            self.root_path = self.root_path + f"-resized-{scale}"
        assert  os.path.exists(self.root_path), f'dataset not found {self.root_path}'

        self.loader = loader

        self.imgs = []


        self.transform = self.__class__.transform()
        self.img_transform = img_transform 
        self.target_transform = LabelToLongTensor()
        self.joint_transform = joint_transform

        self.path_dict = {
            "train": self.root_path+'/semanticSegmentationTrainSet',
            "val": self.root_path+'/semanticSegmentationValidationSet',
            "test": self.root_path+'/semanticSegmentationTestSet'
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

        if self.img_transform is not None:
            img = self.img_transform(img)

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

