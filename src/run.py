import yaml
from pathlib import Path
import json
import sys
import importlib 
import logging
import shutil
import torch
# Prevent "No handler found" warnings and keep logging silent unless configured by the user
logging.getLogger(__name__).addHandler(logging.NullHandler())


root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

from src.tyaml import Tyaml
from src.csv_logger import CSVLogger
from src.settings import load_settings


class Run:
    """
    Class to manage the run configuration, data, and results.
    It initializes the run with a name, sets up folders for results and data,
    and provides methods for saving and loading objects, logging metrics, and clearing data.
    """

    @classmethod
    def _load_and_validate_settings(cls):
        settings = load_settings()
        required_keys = ["run_folder", "data_folder"]
        for key in required_keys:
            if key not in settings:
                raise KeyError(f"Missing required setting '{key}'")
        return settings


    def __init__(self, name=None, track=True):

        self.settings = self._load_and_validate_settings()

        if name is None:
            self.name = Path.cwd().name
        else:
            if not isinstance(name, str):
                raise TypeError("name must be a string")
            self.name = name
        logging.info(f"Initializing Run with name: {self.name}")

        self.folder = Path(self.settings['run_folder']) / self.name
        if not self.folder.exists():
            raise FileNotFoundError(f"Run folder {self.folder} not found")

        self.results_folder = self.folder / 'results'
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.outputs_folder = self.folder / 'outputs'
        self.outputs_folder.mkdir(parents=True, exist_ok=True)
        self.data_folder = Path(self.settings['data_folder']) / self.name
        self.data_folder.mkdir(parents=True, exist_ok=True)

        self._config = Tyaml(self.folder / 'config.yaml')
        self.config = self._config.data
        self.track = self._config.track

        module = importlib.import_module(f"datasets.{self.config['dataset']}")
        self.Dataset = getattr(module, 'Dataset')


        self.metrics = {}



    def __str__(self):
        return f"Experiment: {self.name}\nConfig: {self.config}"
    

    ##################################################################################
    ''' methods for saving and loading objects (e.g., model, optimizer, scheduler) '''
    ##################################################################################

    def save(self, objectClassName, objectData, epoch):
        """
        Save an object (e.g., model, optimizer, scheduler) to a file.

        ***Checkpoint saving convention:
            - epoch_0: initial state (before any training)
            - epoch_N: state after completing N epochs
            (e.g., epoch_1 = after first epoch, epoch_2 = after second, etc.)
            Ensures consistent tracking and resume capability.
        """
        assert isinstance(objectClassName, str), "objectClassName must be a string"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        fpath = self.data_folder / f"{objectClassName}-{epoch}.pth"  
        toSave = {
            'epoch': epoch,
            'data': objectData,
        }

        # Check if the file already exists
        if fpath.exists():
            # Save to a temporary file and raise an error
            tmp_fpath = self.data_folder / f"tmp-{objectClassName}-{epoch}.pth"
            torch.save(toSave, tmp_fpath)
            raise FileExistsError(f"{objectClassName}-{epoch} file {fpath} already exists. Temporary data saved to {tmp_fpath}")
        else:
            torch.save(toSave, fpath)
    
    def save_weights(self, state_dict, epoch):
        return self.save('weights', state_dict, epoch)

    def get_last_epoch(self, objectClassName):
        """
        Get the last saved object of a specific class.
        """
        assert isinstance(objectClassName, str), "objectClassName must be a string"
        assert self.data_folder.exists(), f'Data folder {self.data_folder} not found'
        
        # Get all files and extract epochs
        files = list(self.data_folder.glob(f'{objectClassName}-*'))
        if not files:
            return 0
        # Extract epochs and find the latest one
        last_epoch = max(int(file.stem.split('-')[1]) for file in files)
        
        # Return the path of the latest file
        return last_epoch
    
    def get_lew(self):
        """
        Get the last epoch for weights.
        """
        return self.get_last_epoch('weights')
    
    def get(self, objectClassName, epoch):
        """
        Get the saved object of a specific class and epoch.
        """
        assert isinstance(objectClassName, str), "objectClassName must be a string"
        fpath = self.data_folder / f'{objectClassName}-{epoch}.pth'
        if not fpath.exists():
            raise FileNotFoundError(f"File {fpath} not found")
        logging.info(f'Loading {objectClassName} at {fpath}')
        return torch.load(fpath)['data']
    
    def get_weights(self, epoch):
        """
        Get the latest saved weights.
        """
        return self.get('weights', epoch)


    ##################################################################################
    ''' methods for saving and loading data '''     
    ##################################################################################


    def new_csv(self, name, header):
        """
        Create a new CSV file with the specified name and header.
        """
        assert isinstance(name, str), "name must be a string"
        name = f'{name}.csv'
        filepath = self.results_folder / name
        self.metrics[name] = CSVLogger(filepath, header=header)

            
    def log_csv(self, name, epoch, data):
        """
        log data to a CSV file with the specified name.
        """
        assert isinstance(name, str), "name must be a string"
        assert type(epoch) is int, "epoch must be an integer"
        assert epoch >= 0, "epoch must be a non-negative integer"

        name = f'{name}.csv'
        if name in self.metrics:
            self.metrics[name].log(epoch, data)
        else:
            raise ValueError(f"CSV file {name} not found in metrics. Please create it first using new_csv method.")



    ##################################################################################
    ''' cleanup '''     
    ##################################################################################

    def clear(self, weights=False):
        """
        Clear all CSV content, history, and log files
        """
        # Clear all files in the results folder
        if self.results_folder.exists() and self.results_folder.is_dir():
            # Remove all files and subfolders in the results folder
            shutil.rmtree(self.results_folder)
            self.results_folder.mkdir(parents=True, exist_ok=True)
            print(f'Removed results: {self.results_folder}')

        # Clear the outputs folder
        if self.outputs_folder.exists() and self.outputs_folder.is_dir():
            shutil.rmtree(self.outputs_folder)
            self.outputs_folder.mkdir(parents=True, exist_ok=True)
            print(f'Removed outputs: {self.outputs_folder}')

        # Clear history.yaml
        self._config.clear()

        # Clear weights if specified
        if weights:
            if self.data_folder.exists():
                shutil.rmtree(self.data_folder)
                self.data_folder.mkdir(parents=True, exist_ok=True)
                print(f'Removed data folder: {self.data_folder}')
 