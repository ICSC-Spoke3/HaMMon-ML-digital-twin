
import sys
from pathlib import Path

# Ensure imports work regardless of the current working directory
# root_dir = Path(__file__).resolve().parents[2]
# sys.path.append(str(root_dir))

import torch
import torch.nn as nn


import torchvision.transforms as transforms

from models import att_unet
from datasets import joint_transforms
from src.run import Run
from src.runner import Runner
from src.patcher import Patcher

from modules.weights import inverse_frequency_weights
from modules.loss_focal import FocalLoss
from modules.loss_dice import DiceLoss


import logging
from src.tools import Tools
# if logging is set to debug level or superior, set DEBUG to True, else False
DEBUG=logging.getLogger().isEnabledFor(logging.DEBUG)
t = Tools(DEBUG)

subfolders = [
              'ConcreteCrack', 
              'Stone331',
              'CrackTree260', 
              'DeepCrack', 
              'CrackLS315',         
              'CrackForest',             
              'CRKWH100', 
              ]  

                                                                                                                            
class Loss(nn.Module):
    def __init__(self, c: float = 0.5, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.c = c
        self.focal = FocalLoss(
            mode='binary',
            alpha=alpha,
            gamma=gamma,
            normalized=True,  # Normalize the loss
            reduction='sum'
        )
        self.last_focal = None
    
        self.dice = DiceLoss(
            mode='binary',
            log_loss=True
        )
        self.last_dice = None
    def forward(self, input, target):

        self.last_focal = self.focal(input, target)
        self.last_dice = self.dice(input, target)
        self.last_loss = self.c * self.last_focal + (1 - self.c) * self.last_dice

        return self.last_loss
    
class TrainRunner(Runner):
    def __init__(self, run: Run, rank: int):
        run.Dataset.stats_from_yaml(run.config["dataset_stats"])  
        super().__init__(run=run, rank=rank)

    ######################################################## Datasets 
    def set_dataset_train(self):
        train_joint_transformer = transforms.Compose([
            joint_transforms.JointRandomHorizontalFlip(),
            joint_transforms.JointRandomRotate90()
            ])
        train_transformer = transforms.Compose([
                transforms.ColorJitter(brightness=self.run.config["brightness"],
                                        contrast=self.run.config["contrast"],
                                        saturation=self.run.config["saturation"],
                                        )
            ])
        print(f"@@@@@@@@@@@@@@@@ Dataset stats: {self.run.Dataset.mean}")
        self.dataset_train = self.run.Dataset(
                            split='train',
                            joint_transform=train_joint_transformer,
                            img_transform=train_transformer,
                            root_path=self.run.config["dataset_folder"],
                            subfolders=subfolders,
        ) 

    def set_dataset_eval(self):
        self.dataset_eval = self.run.Dataset(
                            split='val',
                            subfolders=subfolders,
                            joint_transform=None,
                            root_path=self.run.config["dataset_folder"])

    ######################################################## Model

    def set_model(self):
        self.model = att_unet.sAttU_Net(img_ch=3, output_ch=1)
   
    def init_weights(self):
        att_unet.init_weights(self.model, init_type='kaiming', gain=0.02)

    ######################################################## Optimizer and Scheduler

    def set_optimizer(self):
        self.optimizer  = torch.optim.Adam(params=self.model.parameters(),
                                           lr  = self.run.config["LR"],
                                           betas = tuple(self.run.config["betas"]),
                                           eps = self.run.config["eps"],
                                           weight_decay = self.run.config["weight_decay"],
        )
 

    def set_scheduler(self):  
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                        mode='min', 
                                                        factor=self.run.config["factor"], 
                                                        patience=self.run.config["patience"])

    # ####################################################### Criterion          

        
    def set_criterion(self):
        weights = inverse_frequency_weights(self.dataset_train.pixel_count, beta=self.run.config["beta"])
        weights = weights / weights.sum()  # Normalize weights
        print(f"Class weights: {weights}")

        self.criterion = Loss(
            c=self.run.config["c"],
            alpha=weights[1].item(),
            gamma=self.run.config["gamma"],
        )

    def set_patcher(self):
        self.patcher = None

class TestRunner(Runner):
    def __init__(self, run: Run, rank: int):
        run.Dataset.stats_from_yaml(run.config["dataset_stats"])  
        super().__init__(run=run, rank=rank)

    ######################################################## Datasets 

    def set_dataset_eval(self):
        self.dataset_eval = self.run.Dataset(
                            split='test',
                            subfolders=subfolders,
                            joint_transform=None,
                            root_path=self.run.config["dataset_folder"])


    ######################################################## Model

    def set_model(self):
            self.model = att_unet.sAttU_Net(img_ch=3, output_ch=1)
                  

        
    def set_criterion(self):
        weights = inverse_frequency_weights(self.dataset_train.pixel_count, beta=self.run.config["beta"])
        weights = weights / weights.sum()  # Normalize weights
        print(f"Class weights: {weights}")

        self.criterion = Loss(
            c=self.run.config["c"],
            alpha=weights[1].item(),
            gamma=self.run.config["gamma"],
        )
    
    ######################################################## Patcher

    def set_patcher(self):
        self.patcher = None
        
#     # def set_patcher(self):
#     #     self.patcher = Patcher(
#     #         model=self.model,
#     #         kernel=tuple(self.run.config["kernel"]),
#     #         stride=tuple(self.run.config["stride"]),
#     #         device=self.rank
#     #     )
