import sys
from pathlib import Path
import yaml
import importlib 
import hashlib
import shutil
import json

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
 
import torch


class Experiment:
    def __init__(self, name=None, track=True):
 
        self.settings_path = root_dir / "settings.yaml"
        assert self.settings_path.is_file(), 'settings.yaml not found'
        with open(self.settings_path, "r") as f:
            settings = yaml.safe_load(f)

        # If name is not provided, extract it from config.yaml in the current working directory
        if name is None:
            self.name = Path.cwd().name
        else:
            assert isinstance(name, str), 'name must be a string'
            self.name = name
  
        self.exp_folder = root_dir / Path(settings['scripts_root_folder']) / self.name
        assert self.exp_folder.exists(), 'exp folder not found'

        self.config_path = self.exp_folder / 'config.yaml'
        assert self.config_path.is_file(), 'experiment config.yaml not found'

        self.weights_folder =  root_dir / Path(settings['weights_root_folder']) / self.name
        self.results_folder =  root_dir / Path(settings['results_root_folder']) / self.name / 'results'
        self.scripts_folder =  root_dir / Path(settings['scripts_root_folder']) / self.name
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.weights_folder.mkdir(parents=True, exist_ok=True)
        self.scripts_folder.mkdir(parents=True, exist_ok=True)
       
        # yn = '' if settings["current_experiment"] == self.name else 'not '
        # print(f'warning: this is {yn}the current experiment')
        if settings["current_experiment"] != self.name:
            print('warning: this is not the current experiment')


        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        
        module = importlib.import_module(f"datasets.{self.config['dataset']}")
        self.Dataset = getattr(module, 'Dataset')

        self.create_csv('train')
        self.create_csv('val')

        if (track):
            self.versioning()

    def create_csv(self, filename):
        results_file_path = self.results_folder / f"results_{filename}.csv"
        
        if not results_file_path.exists():
            with results_file_path.open(mode='a') as file:
                # Check the length of class_names to determine the header format
                if len(self.Dataset.class_names) == 1:
                    header = "Epoch,Time,Loss,Thresholds,Accuracy,Precision,Recall,Dice,IoU\n"
                else:
                    header = "Epoch,Time,Loss,Error," + ",".join(self.Dataset.class_names) + "\n"
                
                file.write(header)

    def versioning(self):
        # 1. Check the last epoch number
        last_epoch = self.get_last_epoch_in_results('train')
        if last_epoch is None:
            last_epoch = 0

        # 2. Calculate the hash of config.yaml
        with open(self.config_path, 'rb') as f:
            config_bytes = f.read()
            config_hash = hashlib.sha256(config_bytes).hexdigest()

        # 3. Load history from history.yaml as an ordered list
        history_path = self.scripts_folder / 'history.yaml'
        history = []

        if history_path.exists():
            with open(history_path, 'r') as f:
                history = yaml.safe_load(f) or []

        # 4. Check if the last entry hash matches the current hash
        if history:
            last_entry = history[-1]
            last_hash = last_entry['hash']
        else:
            last_hash = None

        # 5. Append new entry if hash is different
        if last_hash != config_hash:
            new_entry = {
                'hash': config_hash,
                'epoch': last_epoch,
                'config': self.config
            }
            history.append(new_entry)
            with open(history_path, 'w') as f:
                yaml.safe_dump(history, f)

        #print(f"Config version: epoch {last_epoch}, hash {config_hash}")


    def __str__(self):
        return f"Experiment: {self.name}\nConfig: {self.config}"
    
    def save_results(self, subset, epoch, time_elapsed, loss, error, results):
        assert self.results_folder.exists(), 'results folder not found'
        csv_file_path = self.results_folder / f'results_{subset}.csv'
        self.create_csv(subset)

        with open(csv_file_path, mode='a') as csv_file:
            if len(self.Dataset.class_names) == 1:
                results_str = ','.join(map(str, results))
                csv_file.write(f"{epoch},{time_elapsed},{loss},{results_str}\n")
            else:
                transf = lambda x: str(float(x))
                results_str = ','.join(map(transf, results))
                csv_file.write(f"{epoch},{time_elapsed},{loss},{error},{results_str}\n")

    def save_epoch_data(self, epoch, data_name, data_dict):
        assert isinstance(data_name, str), "data_name must be a string"
        assert " " not in data_name, "data_name must not contain spaces"
        assert self.results_folder.exists(), 'results folder not found'
        csv_file_path = self.results_folder / f'epoch_{data_name}.csv'
        json_data = json.dumps(data_dict).replace('"', "'")  # Evita problemi di parsing
                
        # Check if CSV exists, if not, create and write header
        file_exists = csv_file_path.exists()
        
        with csv_file_path.open(mode='a+', encoding='utf-8') as file:

            if not file_exists:
                header = 'Epoch,Data\n'
                file.write(header)

            file.write(f'{epoch},"{json_data}"\n')

            




    # def save_results(self, subset, epoch, time_elapsed, loss, error, results):
    #     #assert subset in ['train', 'val', 'test'], "subset must be 'training', 'val', or 'test'"
    #     assert self.results_folder.exists(), 'results folder not found'
    #     csv_file_path = self.results_folder / f'results_{subset}.csv'
    #     self.create_csv(subset)
 
    #     transf = lambda x : str(float(x))
    #     with open(csv_file_path, mode='a') as csv_file:
    #         results_str = ','.join(map(transf, results))
    #         csv_file.write(f"{epoch},{time_elapsed},{loss},{error},{results_str}\n")

        
    def save_weigths(self, state_dict, startEpoch):
        self.weights_folder.mkdir(parents=True, exist_ok=True)
        weights_fpath = self.weights_folder / f"weights-{startEpoch}.pth"  
        toSave = {
            'startEpoch': startEpoch,
            'state_dict': state_dict,
        }

        # Check if the weights file already exi            'scheduler':schedulersts
        if weights_fpath.exists():
            # Save to a temporary file and raise an error
            tmp_weights_fpath = self.weights_folder / f"tmp-{startEpoch}.pth"
            # Assuming model has a method to save itself to a path
            torch.save(toSave, tmp_weights_fpath)
            raise FileExistsError(f"Weights file {weights_fpath} already exists. Temporary weights saved to {tmp_weights_fpath}")
        else:
            torch.save(toSave, weights_fpath)

    def load_weights(self, model, epoch):
        fpath = self.weights_folder / f'weights-{epoch}.pth'
        assert fpath.exists(), f'weights file {fpath} not found'
        weights = torch.load(fpath)
        startEpoch = weights['startEpoch']
        model.load_state_dict(weights['state_dict'])
        print(f'loaded weights epoch: {startEpoch}')
 
    def load_from_checkpoint(self, checkpoint_filename, model, exclude=None):
        exclude = [] if exclude is None else exclude

        assert checkpoint_filename.exists(), f'weights file {checkpoint_filename} not found'
        weights = torch.load(checkpoint_filename)

        for layer in exclude:
            del weights['state_dict'][f'{layer}.weight']
            del weights['state_dict'][f'{layer}.bias']

        # load weights ignore missing layers
        model.load_state_dict(weights['state_dict'], strict=False)


    
    def get_last_weights(self):
        assert self.weights_folder.exists(), 'weights folder not found'        
        # Get all weight files and extract epochs
        weight_files = list(self.weights_folder.glob('weights-*'))
        if not weight_files:
            return None
        # Extract epochs and find the latest one
        latest_epoch = max(int(file.stem.split('-')[1]) for file in weight_files)
        # Return the path of the latest weights file
        return (latest_epoch, self.weights_folder / f'weights-{latest_epoch}')
    
    def load_latest_weights(self, model):
        latest_epoch = self.get_last_weights()[0]
        self.load_weights(model, latest_epoch)

    def get_last_epoch_in_results(self, subset):
        #assert subset in ['train', 'val', 'test'], "subset must be 'train', 'val', or 'test'"
        csv_file_path = self.results_folder / f'results_{subset}.csv'
        assert csv_file_path.exists(), f'CSV file {csv_file_path} not found'
        
        # Read the CSV file and find the last epoch, regardless of ordering
        last_epoch = None
        with open(csv_file_path, mode='r') as csv_file:
            next(csv_file)  # Skip header
            epochs = []
            for line in csv_file:
                try:
                    epoch = int(line.split(',')[0])
                    epochs.append(epoch)
                except ValueError:
                    raise ValueError(f"Malformed line detected: {line.strip()}")
        
        if epochs:
            last_epoch = max(epochs)
    
        return last_epoch



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

        # Clear the scripts_folder/logs folder
        logs_folder = self.scripts_folder / 'logs'
        if logs_folder.exists() and logs_folder.is_dir():
            shutil.rmtree(logs_folder)
            logs_folder.mkdir(parents=True, exist_ok=True)
            print(f'Removed logs: {logs_folder}')

        # Clear history.yaml
        history_path = self.scripts_folder / 'history.yaml'
        if history_path.exists():
            history_path.unlink()
            print(f'Cleared history file: {history_path}')
        else:
            print('No history file found to clear.')

        # Clear weights if specified
        if weights and self.weights_folder.exists() and self.weights_folder.is_dir():
            shutil.rmtree(self.weights_folder)
            self.weights_folder.mkdir(parents=True, exist_ok=True)
            print(f'Removed weights: {self.weights_folder}')

    def save_scheduler(self, epoch, scheduler):
        assert self.weights_folder.exists(), f'weights folder {self.weights_folder} not found'
        scheduler_fpath = self.weights_folder / f"scheduler-{epoch}.pth"
        torch.save(scheduler.state_dict(), scheduler_fpath)
        print(f'Scheduler state saved for epoch {epoch} at {scheduler_fpath}')

    def load_scheduler(self, epoch, scheduler):
        scheduler_fpath = self.weights_folder / f"scheduler-{epoch}.pth"
        if scheduler_fpath.exists():
            scheduler.load_state_dict(torch.load(scheduler_fpath))
            print(f'Scheduler state loaded for epoch {epoch} from {scheduler_fpath}')
        else:
            print('no scheduler state found')
            return

    def save(self, whatName, startEpoch, whatData):
        assert isinstance(whatName, str), f"whatName must be a string, but got {type(whatName).__name__}"
        self.weights_folder.mkdir(parents=True, exist_ok=True)
        fpath = self.weights_folder / f"{whatName}-{startEpoch}.pth"  
        toSave = {
            'startEpoch': startEpoch,
            whatName: whatData,
        }

        # Check if the weights file already exi            'scheduler':schedulersts
        if fpath.exists():
            # Save to a temporary file and raise an error
            tmp_fpath = self.weights_folder / f"tmp-{whatName}-{startEpoch}.pth"
            # Assuming model has a method to save itself to a path
            torch.save(toSave, tmp_fpath)
            raise FileExistsError(f"{whatName}-{startEpoch} file {fpath} already exists. Temporary weights saved to {tmp_fpath}")
        else:
            torch.save(toSave, fpath)

    def get_last_epoch(self, whatName):
        assert isinstance(whatName, str), f"whatName must be a string, but got {type(whatName).__name__}"
        assert self.weights_folder.exists(), 'weights folder not found'        

        # Get all files and extract epochs
        files = list(self.weights_folder.glob(f'{whatName}-*'))
        if not files:
            return None
        # Extract epochs and find the latest one
        last_epoch = max(int(file.stem.split('-')[1]) for file in files)
        # Return the path of the latest weights file
        return last_epoch
    
    def get_latest_file(self, whatName):
        last_epoch = self.get_last_epoch(whatName)

        # Check if last_epoch is None
        if last_epoch is None:
            return None  
        
        # Construct the path to the latest weights file
        latest_file = self.weights_folder / f"{whatName}-{last_epoch}.pth"
        
        return latest_file
    
    def load_latest(self, whatName):
        latest_file = self.get_latest_file(whatName)
        return torch.load(latest_file)