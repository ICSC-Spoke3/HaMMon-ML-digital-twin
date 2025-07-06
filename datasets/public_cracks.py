
import os
from pathlib import Path
import yaml
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
#from torchvision.datasets.folder import is_image_file, default_loader
from torchvision import transforms
import logging

settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"

if settings_path.exists():
    with settings_path.open('r') as f:
        settings = yaml.safe_load(f)
    datasets_folder = Path(settings.get('datasets_folder'))
else: 
    datasets_folder = None



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
    
class LabelToFloatTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return torch.from_numpy(pic).float()
        else:
            return torch.from_numpy(np.array(pic)).float()
        
class DivideBy255(object):
    """NOTE:
    This transform assumes that the label masks contain only values 0 (for background) and 255 (for crack).
    The division by 255 is used to quickly normalize the values to 0.0 and 1.0 for performance reasons.
    No validation is performed here: if the mask contains unexpected values (e.g., 128), the result will be incorrect.
    Make sure that value validation (e.g., checking that only 0 and 255 are present) is done earlier during preprocessing.
    """
    def __call__(self, pic):
        """
        Converts a PIL Image with pixel values in the range [0, 255] to [0, 1].
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            pic = pic / 255.0
        else:
            pic = np.array(pic) / 255.0
        return pic


class Dataset(data.Dataset):
    
    # classes
    class_names = class_names
    class_colors = class_colors
    # stats
    mean = mean
    std = std
    # Normalized RGB stats (scaled to [0, 1])
    norm_mean = [m / 255.0 for m in mean]
    norm_std = [s / 255.0 for s in std]
    image_count = image_count
    pixel_count = pixel_count

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
    def target_transform(cls):
        """
        Target transform: converts label masks to float tensors and normalizes pixel values to [0, 1].
        """
        return transforms.Compose([
            DivideBy255(),  # Convert to [0, 1] range
            #LabelToLongTensor()
            LabelToFloatTensor()  # Convert to float tensor
        ]) # Choose the target transform depending on the Loss Function:
    
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
                 subfolders=None,
                 img_transform=None,
                 joint_transform=None,
                 root_path=None,
                ):
        
 

        if root_path is None:
            assert datasets_folder is not None, "datasets_folder is not defined. Please set the datasets_folder in settings.yaml."
            self.root_path = datasets_folder / 'public-cracks'
        else:
            root_path = Path(root_path)
            if root_path.is_absolute():
                self.root_path = root_path
            else:
                assert datasets_folder is not None, "datasets_folder is not defined. Please set the datasets_folder in settings.yaml."
                self.root_path = datasets_folder / root_path


        assert  os.path.exists(self.root_path), f'dataset not found {self.root_path}'

        assert split in ('train', 'val', 'test')

        if subfolders is None: 
            subfolders = [item.name for item in self.root_path.iterdir() if item.is_dir()]
        assert isinstance(subfolders, list) and len(subfolders) > 0, "subfolders must be a non-empty list"

        self.subfolders = subfolders
        self.split = split


        #self.loader = loader

        self.imgs = []

        self.img_transform = img_transform 
        self._transform = self.__class__.transform()
        self._target_transform = self.__class__.target_transform()
        self.joint_transform = joint_transform

        self.fulfill()
    
    def _get_path(self, index):
        subfolder, fname = self.imgs[index]
        img_path = self.root_path / f'{subfolder}/{self.split}/imgs/{fname}.jpg'
        target_path = self.root_path / f'{subfolder}/{self.split}/labels/{fname}.png'
        return img_path, target_path

    def __getitem__(self, index):

        img_path, target_path = self._get_path(index) 
        
        img = self.__class__.load_img(img_path)
        target = self.__class__.load_target(target_path)

        if self.joint_transform is not None: 
            img, target = self.joint_transform([img, target])

        if self.img_transform is not None:
            img = self.img_transform(img)

        img = self._transform(img)
        target = self._target_transform(target)

        return img, target


    def __len__(self):
        return len(self.imgs)


    def fulfill(self):
        """
        Fills the dataset with images and their corresponding labels.
        """
        for subfolder in self.subfolders:
            dir = self.root_path / subfolder / self.split / 'imgs'
            assert dir.exists(), f"Subfolder {dir} doesn't exist"


            for file_path in sorted(dir.rglob('*.jpg')):
                if not file_path.suffix == '.jpg':
                    raise ValueError(f"Expected .jpg files, but found {file_path.name} in {file_path.parent}")
                self.imgs.append((subfolder,file_path.name[:-4]))  # Remove the '.jpg' extension

        logging.info(f"Dataset loaded with {len(self.imgs)} images.")