import yaml
from pathlib import Path
import json
import sys
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

from src.run import Run

class Plotter:
    def __init__(self, run: str):

        assert isinstance(run, str), "Run must be a string"
        self.run = Run(run)

        self.metrics_header = self.get_metrics_header()
        self.dataframes = {}

        

    #################################### Pandas helpers

    def _pd(self, name):
        """
        Loads a pandas DataFrame from a CSV file in the run's results folder."""
        csv_path = self.run.results_folder / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file {csv_path} does not exist")
        else:
            df = pd.read_csv(csv_path)
            self.dataframes[name] = df
            return df
        
    def pd(self, name):
        """
        Returns a pandas DataFrame for the specified CSV file.
        If the DataFrame is already loaded, it returns the cached version.
        """
        if name not in self.dataframes:
            return self._pd(name)
        else:
            return self.dataframes[name]
        
    
    def combine(self, dfs, column):
        # Create a new DataFrame using 'Epoch' from the first DataFrame as index
        combined = pd.DataFrame()

        for name, df in dfs.items():
            if 'Epoch' not in df.columns or column not in df.columns:
                raise ValueError(f"Both 'Epoch' and '{column}' must be present in DataFrame '{name}'")
            
            df = df.set_index('Epoch')
            combined[name] = df[column]

        # Reset index to have 'Epoch' as a column
        combined = combined.reset_index()
        return combined
    

    def get_metrics_header(self):
        metrics_path = self.run.results_folder / "metrics_train.csv"
        df = pd.read_csv(metrics_path, nrows=0)  
        columns = df.columns.tolist()
        return [col for col in columns if col != "Epoch"]
        

    #################################### Plot helpers


    # def _plot(self, df, title, xlabel="Epoch", figsize=(10, 6), epoch=-1, save=False):
    #     """
    #     Plots the DataFrame with specified title and labels.
    #     If save is True, saves the plot to the outputs folder.
    #     """
    #     if epoch != -1:
    #         df = df[df['Epoch'] <= epoch]
    #     plt.figure(figsize=figsize)
    #     for col in df.columns:
    #         if col != 'Epoch':
    #             plt.plot(df['Epoch'], df[col], label=col)

    #     plt.xlabel(xlabel)
    #     plt.title(title)
    #     plt.legend()
    #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #     plt.tight_layout()

    #     if save:
    #         save_path = self.run.outputs_folder / f"{title.replace(' ', '_')}.png"
    #         plt.savefig(save_path)
    #         print(f"Plot saved to {save_path}")

    #     plt.show()
    def _plot(self, df, title, xlabel="Epoch", figsize=(10, 6), epoch=-1, save=False):
        """
        Plots the DataFrame with specified title and labels.
        If save is True, saves the plot to the outputs folder.
        """
        # Elenco fisso di colori da usare ciclicamente
        fixed_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        if epoch != -1:
            df = df[df['Epoch'] <= epoch]
        plt.figure(figsize=figsize)
        color_index = 0
        for col in df.columns:
            if col != 'Epoch':
                color = fixed_colors[color_index % len(fixed_colors)]
                plt.plot(df['Epoch'], df[col], label=col, color=color)
                color_index += 1

        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend()
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if save:
            save_path = self.run.outputs_folder / f"{title.replace(' ', '_')}.png"
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()



    #################################### Metrics and IoU



    def plot(self, name, columns=None, figsize=(10, 6), epoch=-1, save=False):
        """
        Plots specified columns (or all columns if None) against 'Epoch'
        from a CSV in the run's results folder.
        """
        df = self.pd(name)
        if df.empty:
            raise ValueError(f"CSV {name} is empty.")

        plot_columns = columns or [col for col in df.columns if col != 'Epoch']

        _df = df[['Epoch'] + plot_columns]

        self._plot(df=_df, title=name, xlabel="Epoch", figsize=figsize, epoch=epoch, save=save)

    def which_csv(self, column):
        """
        Returns the setting (metrics or IoU) based on the column name.
        """
        if column in self.metrics_header:
            return 'metrics'
        elif column in self.run.Dataset.class_names:
            return 'IoU'
        else:
            raise ValueError(f"Column '{column}' is not a valid metric or class name.")


    def plot_train_val(self, column, epoch=-1, save=False):
        """
        Plots the specified column for both training and validation datasets.
        """
        
        train_df = self.pd(f'{self.which_csv(column)}_train')
        val_df = self.pd(f'{self.which_csv(column)}_val')

        if train_df.empty or val_df.empty:
            raise ValueError("Train or validation DataFrame is empty.")

        combined = self.combine({'train': train_df, 'val': val_df}, column)

        self._plot(combined, title=f"{column}", xlabel="Epoch", epoch=epoch, save=save)

    def plot_runs(self, runs, column, set_, epoch=50, save=False):
        """
        Plots the specified column for multiple runs.
        """
        if isinstance(runs, str):
            runs = [runs]
        
        if not set_ in ['train', 'val', 'test']:
            raise ValueError("set_ must be one of 'train', 'val', or 'test'.")

        dfs = {}
        for run in runs:
            plotter = Plotter(run)
            df = plotter.pd(f'{self.which_csv(column)}_{set_}')
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in run {run}.")
            dfs[run] = df[['Epoch', column]]

        combined = self.combine(dfs, column)

        self._plot(combined, title=f"{column} across runs", xlabel="Epoch", epoch=epoch, save=save)

    @property
    def sorted_classes(self):
        """
        Returns the class names sorted by pixel count in ascending order.
        """

        class_names, pixel_count = self.run.Dataset.class_names, self.run.Dataset.pixel_count

        l = dict(zip(class_names, pixel_count))
        l = dict(sorted(l.items(), key=lambda item: item[1]))

        return reversed(l.keys())

    def plot_sorted_IoU(self, epoch=-1,save=False):
        l = self.sorted_classes
        for label in l:
            self.plot_train_val(label, epoch=epoch, save=save)



    #################################### Confusion Matrix 

    def _cm_to_np(self, name: str, epoch: int=-1):
        """
        Returns the confusion matrix DataFrame for the specified run.
        """
        df = self.pd(f"CM_{name}")
        if df.empty:
            raise ValueError(f"Confusion matrix for {name} is empty")

        cm_flat = df.iloc[epoch-1].values[1:]  # extract a single row

        num_classes = len(self.run.Dataset.class_names)

        return np.array(cm_flat).reshape((num_classes, num_classes))
    
    def cm(self, name: str, epoch: int=-1, save: bool=False, percent: bool=True):
        """
        visualize the cm as a heatmap.
        If save is True, it saves the heatmap to the outputs folder.
        """
        cm_np = self._cm_to_np(name, epoch)
        if cm_np.size == 0:
            raise ValueError(f"Confusion matrix for {name} is empty")
        

        if percent:
            cm_np = cm_np / cm_np.sum(axis=1, keepdims=True) * 100
            # cm_np = np.round(cm_np, 2)

        class_names = self.run.Dataset.class_names

        plt.figure(figsize=(12, 9))
        sns.heatmap(cm_np, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch if epoch >= 0 else "Last"}')
        plt.tight_layout()

        if save:
            plt.savefig(self.run.outputs_folder / f"cm_{name}_{epoch}.png")

        plt.show()

    #################################### Pytorch Objects Plotter

    def torch_to_pd(self, objectClassName, save: bool=False):
        """
        Converts a saved PyTorch object to a pandas DataFrame.
        The object should be a dictionary with keys as metric names and values as lists or single values.
        If save is True, it saves the DataFrame to the results folder as a csv file."""

        def filter(obj):
            if not isinstance(obj, dict):
                raise TypeError(f'Expected dict, got {type(obj)}')
            out = {}
            for k,v in obj.items():
                if isinstance(v, int) or isinstance(v, float):
                    out[k] = v
                elif isinstance(v, list):
                    if len(v) == 1:
                        out[k] = v[0]
            return out
        
        le = self.run.get_last_epoch(objectClassName)

        if le == 0:
            raise ValueError(f'No epochs found for {objectClassName}')
        records = []
        for epoch in range(1, le + 1):
            data = self.run.get(objectClassName, epoch)
            data = filter(data)
            data['Epoch'] = epoch
            records.append(data)
        df =  pd.DataFrame(records)

        if save:
            save_path = self.run.results_folder / f"{objectClassName}.csv"
            df.to_csv(save_path, index=False)

        return df

        


        
