import os
from pathlib import Path
import yaml
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms



                ################
                ### FloodNet ###
                ################

"""
FloodNet is a high-resolution UAV dataset for post-disaster scene understanding,
collected after Hurricane Harvey. It includes pixel-wise annotations for semantic segmentation,
as well as labels for classification and visual question answering (VQA).
Paper: https://doi.org/10.1109/ACCESS.2021.3090981
Dataset (Dropbox): https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=0

Expected layout:

{datasets_folder}/FloodNet[-resized-{SCALE}]/FloodNet-Supervised_v1.0/train/-org-img/*.jpg
matching masks under train/-label-img/*_lab.png
same pattern for val/ and test/ (each with -org-img images and -label-img masks).

If you want to use smaller resolutions, resize both images and masks ahead of time,
place the results under FloodNet-resized-{SCALE} at the dataset root, and keep the standard subfolders


"""

settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"

if settings_path.exists():
    with settings_path.open('r') as f:
        settings = yaml.safe_load(f)
    datasets_folder = Path(settings.get('datasets_folder'))
else: 
    datasets_folder = None

DATASET_PATH = f"{datasets_folder}/FloodNet"


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


    @classmethod
    def stats_from_yaml(cls, path):

        if path is None:
            raise ValueError("Path to stats file must be defined. Please set the path in settings.yaml.")
        else:
            path = Path(path)
            if not path.is_absolute():
                if datasets_folder is None:
                    raise ValueError("datasets_folder is not defined. Please set the datasets_folder in settings.yaml.")
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

    def __init__(self, *,
                 split,
                 scale,
                 img_transform=None,
                 joint_transform=None,
                 loader=default_loader):
        assert split in ('train', 'val', 'test')

        self.split = split

        if scale is not in (None, 3000):
            self.root_path = self.root_path + f"-resized-{scale}/FloodNet-Supervised_v1.0"
        else:
            self.root_path = self.root_path + "/FloodNet-Supervised_v1.0"
        assert  os.path.exists(self.root_path), f'dataset not found {self.root_path}'

        self.loader = loader

        self.imgs = []


        self.transform = self.__class__.transform()
        self.img_transform = img_transform 
        self.target_transform = LabelToLongTensor()
        self.joint_transform = joint_transform

        self.path_dict = {
            "train": self.root_path+'/train',
            "val": self.root_path+'/val',
            "test": self.root_path+'/test'
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




