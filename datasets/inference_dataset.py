from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.from_numpy(np.array(pic, dtype=np.int64))
        return label


class InferenceDataset(data.Dataset):
   
    def __init__(self, Dataset_model, dataset_folder):

        assert issubclass(Dataset_model, data.Dataset), "Dataset_model must be a subclass of torch.utils.data.Dataset"       
        self.Dataset_model = Dataset_model

        self.dataset_folder = Path(dataset_folder)
        if not self.dataset_folder.exists() or not self.dataset_folder.is_dir():
            raise ValueError(f"Dataset folder '{self.dataset_folder}' does not exist or is not a directory.")        
        
        self.loader = default_loader

        self.imgs = []

        self.norm_mean=[m/255.0 for m in self.Dataset_model.mean]
        self.norm_std=[s/255.0 for s in self.Dataset_model.std]
        self.class_names = self.Dataset_model.class_names
        self.class_colors = self.Dataset_model.class_colors

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        self.target_transform = LabelToLongTensor()


        self._add_to_dataset(self.dataset_folder)

    

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        img = ImageOps.exif_transpose(img) 
        img = self.transform(img)
        return img


    def __len__(self):
        return len(self.imgs)
    
    def _add_to_dataset(self, dir):
        dir_path = Path(dir)
        for file in dir_path.rglob("*"):
            if file.is_file() and is_image_file(file.name):
                self.imgs.append(file)

