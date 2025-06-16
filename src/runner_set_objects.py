
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from src.settings import load_settings


class SetObjects:
    def __init__(self, run, rank):
        self.run = run
        self.rank = rank
        #self.lew = self.run.get_lew()
        self.settings = load_settings()

        if not hasattr(self, 'set_dataset_train'):
            self.dataset_train = None
        else:        
           self.set_dataset_train()

        if not hasattr(self, 'set_dataset_eval'):
            self.dataset_eval = None    
        else:
          self.set_dataset_eval()
          
        self._set_dataloader_train()
        self._set_dataloader_eval()

        self._set_model()

        if self.dataset_train is not None:
            self._set_optimizer()
            self._set_scheduler()
        
        self._set_criterion()

        self.set_patcher()




    def _set_dataloader_train(self):
    
        if not hasattr(self, 'dataset_train') or self.dataset_train is None:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            self.dataset_train = None
        else:
            self.dataloader_train = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.run.config["batch_size"],
                shuffle=False,
                num_workers=self.settings["num_workers"],
                pin_memory=True,
                sampler=DistributedSampler(self.dataset_train) 
            )

    def _set_dataloader_eval(self):
        if not hasattr(self, 'dataset_eval') or self.dataset_eval is None:
            self.dataset_eval = None
        else:
            self.dataloader_eval = torch.utils.data.DataLoader(
                self.dataset_eval,
                batch_size=self.run.config["batch_size_eval"],
                shuffle=False,
                num_workers=self.settings["num_workers"],
                pin_memory=True,
                sampler=DistributedSampler(self.dataset_eval) 
            )  

    def _set_model(self, epoch=None):
        """
        Initializes the model, loads weights if available, and wraps it in DDP.
        """

        self.set_model()

        if self.model is None:
            raise ValueError("Model must be set before initializing weights or loading state.")
        
        self.model.to(self.rank)

        epoch = self.lew if epoch is None else epoch

        if epoch == 0: 
            logging.info('Initializing weights from scratch')
            if not hasattr(self, 'init_weights'):
                raise ValueError("init_weights method must be defined to initialize model weights.")
            self.init_weights()
        elif epoch > 0:
            weights = self.run.get_weights(epoch)
            self.model.load_state_dict(weights)
            logging.info(f'Loaded weights from epoch {epoch}')
        else:
            raise ValueError(f'Invalid last epoch: {epoch}. It must be 0 or greater.')
        
        # wrap the model in DDP
        self.model = DDP(self.model, device_ids=[self.rank])
        

    def _set_optimizer(self):
        """
        initializes the optimizer with the model parameters 
        and load its saved state dict
        """
        assert hasattr(self, 'model'), "Model must be set before setting the optimizer." 

        self.set_optimizer()
 
        if self.lew > 0:
            opt = self.run.get('optimizer', self.lew)
            logging.info(f'Loaded optimizer from epoch {self.lew}')
            self.optimizer.load_state_dict(opt) 

    def _set_scheduler(self):
        """
        Sets the learning rate scheduler
        and loads its saved state dict.
        """
        assert hasattr(self, 'optimizer'), "Optimizer must be set before setting the scheduler."

        self.set_scheduler()

        if self.lew > 0:
            sch = self.run.get('scheduler', self.lew)
            logging.info(f'Loaded scheduler from epoch {self.lew}')
            self.scheduler.load_state_dict(sch)

    def _set_criterion(self):
        """
        Sets the loss function.
        """
        if hasattr(self, 'class_weights'):
            self.class_weights.to(self.rank)  # Ensure class weights are on the correct device

        self.set_criterion()
        self.criterion.to(self.rank)

    # def set_patcher(self, patcher_func):
    #     """
    #     Sets the patcher function for data augmentation.
    #     """
    #     self.patcher = patcher_func(self.model,self.rank)



