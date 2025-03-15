import os
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms


                ##################################
                ### Crack Segmentation Dataset ###
                ##################################

"""
This script defines a PyTorch dataset class for the "Crack Segmentation Dataset" (CDS) from Kaggle 
(https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset).
The dataset is designed for semantic segmentation tasks, providing images of surfaces with cracks 
and corresponding masks. 

The dataset comprises approximately 11,200 images, created by merging and resizing 12 public crack segmentation datasets.

CDS was divided into "train" and "test" sets, maintaining the proportions of the original datasets.
Here, the "test" set was further split into "test" and "validation" sets, preserving the original proportions.
"""


DATASET_PATH = "/datasets/kaggle-crack_segmentation_dataset"


class_names = ['crack']
class_colors = [
    (255, 0, 0)       # crack
]
# class_names = ['background', 'crack']
# class_colors = [
#     (0, 0, 0),          # background
#     (255, 0, 0),        # crack
# ]

# RGB STATS
mean = [120.90263287390626, 125.89930617298026, 128.71869615485824]
std = [41.32545473300588, 39.064575180017655, 38.833432965844274]

# NUMBER OF IMAGES EACH LABEL APPEARS IN:
image_count = None
# NUMBER OF PIXELS FOR EACH LABEL:
pixel_count = None


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.from_numpy(np.array(pic, dtype=np.int64))
        return label

class LabelToFloatTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).float()  # Cambia da long a float
        else:
            label = torch.from_numpy(np.array(pic, dtype=np.float32))  # Usa float32

        # label = torch.where(label<128, 0,1)
        label = torch.where(label<128, torch.tensor(0.0), torch.tensor(1.0))
        
        return label





class Dataset(data.Dataset):
    print('ricordarsi di sistemare il dataset (masks to 0 and 1) prima di lanciare training lunghi')
    root_path = DATASET_PATH
    
    # classes
    class_names = class_names
    class_colors = class_colors
    # stats
    mean = mean
    std = std

    norm_mean = [m / 255.0 for m in mean]
    norm_std = [s / 255.0 for s in std]

    image_count = image_count
    pixel_count = pixel_count

    @classmethod
    def Compose(cls):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.norm_mean, std=cls.norm_std)
        ])
    
    @classmethod
    def Loader(cls):
        return default_loader


    def __init__(self, *,
                 split='train',
                 joint_transform=None,
                 loader=default_loader):
        assert split in ('train', 'val', 'test')

        self.split = split
        
        assert  os.path.exists(self.root_path), f'dataset not found {self.root_path}'

        self.loader = loader if callable(loader) else self.__class__.Loader()

        self.imgs = []

        self.norm_mean=[m/255.0 for m in self.__class__.mean]
        self.norm_std=[s/255.0 for s in self.__class__.std]

        self.transform = self.__class__.Compose()

        self.target_transform = LabelToFloatTensor()
        #self.target_transform = LabelToLongTensor()
        self.joint_transform = joint_transform


        self._add_to_dataset(self.root_path)

    def _get_path(self, index):
        path = self.imgs[index]
        target_path = path.replace('/images/', '/masks/')
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
        split = 'test' if self.split == 'val' else self.split
        dir = f"{dir}/{split}/images"

    
        excluded_datasets = ['Eugen', 'Sylvie', 'Rissbilder', 'Volker', 'cracktree200','GAPS384']

        for root, _, fnames in sorted(os.walk(dir)):
            odd = False
            fnames = sorted(fnames)
            for fname in fnames:

                if any(excluded in fname for excluded in excluded_datasets):
                    #print(f"Excluding file: {fname}")  # Debug per verificare quali file vengono esclusi
                    continue
                
                # print(fname.replace('/datasets/kaggle-crack_segmentation_dataset/train/images/', ''))
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    if (self.split == 'train'):  
                        self.imgs.append(path)
                    elif ((self.split == 'val') and (not odd)):
                        self.imgs.append(path)
                    elif ((self.split == 'test') and odd):
                        self.imgs.append(path)
                else: 
                    raise ValueError(f'{fname} is not image file')
                odd = not odd
                        
