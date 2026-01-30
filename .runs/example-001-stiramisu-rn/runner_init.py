import sys
from pathlib import Path

# Ensure imports work regardless of the current working directory
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

import torch
import torch.nn as nn


import torchvision.transforms as transforms

from models import stiramisu
from datasets import joint_transforms
from src.run import Run
from src.runner import Runner
from modules.weights import norm_enetw
from src.patcher import Patcher

import logging
from src.tools import Tools
# if logging is set to debug level or superior, set DEBUG to True, else False
DEBUG=logging.getLogger().isEnabledFor(logging.DEBUG)
t = Tools(DEBUG)


class TrainRunner(Runner):
    def __init__(self, run: Run, rank: int):
        run.Dataset.stats_from_yaml(run.config["dataset_stats"])
        super().__init__(run, rank)


    ######################################################## Datasets 
    def set_dataset_train(self):

        crop_size = self.run.config["crop_size"]

        train_joint_transformer = transforms.Compose([
                joint_transforms.JointRandomCrop(crop_size),
                joint_transforms.JointRandomFlip(),
                ])
        self.dataset_train = self.run.Dataset(
                            split='train',
                            scale=self.run.config["dataset_scale"],
                            joint_transform=train_joint_transformer
        ) 

    def set_dataset_eval(self):
        crop_size = self.run.config["crop_size"]
        val_joint_transformer = transforms.Compose([
                joint_transforms.FixedUpperLeftCrop(crop_size),
                ])  
        self.dataset_eval = self.run.Dataset(
                            split='val',
                            scale=self.run.config["dataset_scale"],
                            joint_transform=val_joint_transformer
        )

    ######################################################## Model
    def set_model(self):
        self.model = stiramisu.sFCDenseNet103(len(self.run.Dataset.class_names))
   
    def init_weights(self):
        stiramisu.weights_init(self.model)

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
        run.Dataset.stats_from_yaml(run.config["dataset_stats"])
        super().__init__(run, rank)

    ######################################################## Datasets 
    def set_dataset_eval(self):
        crop_size = self.run.config["crop_size"]
        test_joint_transformer = transforms.Compose([
                joint_transforms.FixedUpperLeftCrop(crop_size),
                ])
        self.dataset_eval = self.run.Dataset(
                            split='test',
                            scale=self.run.config["dataset_scale"],
                            joint_transform=test_joint_transformer
        )
 
    ######################################################## Model
    def set_model(self):
        self.model = stiramisu.sFCDenseNet103(len(self.run.Dataset.class_names))

    ######################################################## Criterion          
    def set_criterion(self):
        pixel_count = self.run.Dataset.pixel_count
        self.class_weights = norm_enetw(pixel_count, c=self.run.config["c"])
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)

    ######################################################## Patcher
    def set_patcher(self):
        self.patcher = None
        
#     ######################################################## Patcher
#     def set_patcher(self):
#         self.patcher = Patcher(
#             model=self.model,
#             kernel=tuplize(self.run.config["kernel"]),
#             stride=tuplize(self.run.config["stride"]),
#             device=self.rank
#         )
