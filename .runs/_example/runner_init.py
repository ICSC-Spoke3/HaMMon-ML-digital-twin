
import sys
from pathlib import Path

# Ensure imports work regardless of the current working directory
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

import torch
import torch.nn as nn


import torchvision.transforms as transforms

from models import att_unet
from datasets import joint_transforms
from src.run import Run
from src.runner import Runner
from src.utils import norm_enetw
from src.patcher import Patcher

import logging
from src.tools import Tools
# if logging is set to debug level or superior, set DEBUG to True, else False
DEBUG=logging.getLogger().isEnabledFor(logging.DEBUG)
t = Tools(DEBUG)

tuplize = lambda x: tuple(map(int, x.split(', ')))

class TrainRunner(Runner):
    def __init__(self, run: Run, rank: int):
        super().__init__(run, rank)

    ######################################################## Datasets 
    def set_dataset_train(self):
        train_joint_transformer = transforms.Compose([
            joint_transforms.JointRandomCrop(tuplize(self.run.config["crop_size"])), 
            joint_transforms.JointRandomHorizontalFlip()
            # joint_transforms.JointRandomRotate90(),
            # joint_transforms.JointRandomFlip()
            ])
        self.dataset_train = self.run.Dataset(
                            split='train',
                            scale=self.run.config["dataset_scale"],
                            joint_transform=train_joint_transformer
        ) 

    def set_dataset_eval(self):
        self.dataset_eval = self.run.Dataset(
                            split='val',
                            scale=self.run.config["dataset_scale"],
                            joint_transform=joint_transforms.FixedUpperLeftCrop(
                                tuplize(self.run.config["crop_size_eval"])))

    ######################################################## Model
    def set_model(self):
        self.model = att_unet.sAttU_Net(img_ch=3, output_ch=len(self.run.Dataset.class_names))
   
    def init_weights(self):
        att_unet.init_weights(self.model, init_type='kaiming', gain=0.02)

    ######################################################## Optimizer and Scheduler

    def set_optimizer(self):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), 
                            lr=self.run.config["LR"], 
                            momentum=self.run.config["momentum"], 
                            weight_decay=self.run.config["weight_decay"])

    def set_scheduler(self):  
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                        mode='min', 
                                                        factor=self.run.config["factor"], 
                                                        patience=self.run.config["patience"])

    # ####################################################### Criterion          

    def set_criterion(self):

        pixel_count = self.run.Dataset.pixel_count
        self.class_weights = norm_enetw(pixel_count, c=self.run.config["c"])#.to(self.rank)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)

    def set_patcher(self):
        self.patcher = None



class TestRunner(Runner):
    def __init__(self, run: Run, rank: int):
        super().__init__(run, rank)

    ######################################################## Datasets 
    def set_dataset_eval(self):
        self.dataset_eval = self.run.Dataset(
                            split='test',
                            scale=self.run.config["dataset_scale"],
                            joint_transform=joint_transforms.FixedUpperLeftCrop(
                                tuplize(self.run.config["crop_size_eval"])))
        
    ######################################################## Model
    def set_model(self):
        self.model = att_unet.sAttU_Net(img_ch=3, output_ch=len(self.run.Dataset.class_names))
    
    ######################################################## Criterion          
    def set_criterion(self):
        pixel_count = self.run.Dataset.pixel_count
        self.class_weights = norm_enetw(pixel_count, c=self.run.config["c"]).to(self.rank)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)
    
    ######################################################## Patcher
    def set_patcher(self):
        self.patcher = Patcher(
            model=self.model,
            kernel=tuplize(self.run.config["kernel"]),
            stride=tuplize(self.run.config["stride"]),
            device=self.rank
        )
