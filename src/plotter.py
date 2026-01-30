
"""
Plotter utilities for training runs: load CSV metrics from a Run’s results folder,
cache them as pandas DataFrames, and generate common visualizations (metrics curves,
IoU plots, confusion matrices, and threshold analyses) for training/validation/test.
"""


from pathlib import Path
import sys

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
        self.common_metrics_header = self.get_params_header()
        self.dataframes = {}

        self.multiclass = self.run.config.get('multiclass', True) # defaults to multiclass


        

    #################################### Pandas helpers

    def _pd(self, name):
        """
        Loads a pandas DataFrame from a CSV file in the run's results folder."""
        csv_path = self.run.results_folder / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file {csv_path} does not exist")
        else:
           return pd.read_csv(csv_path)
         
        
    def pd(self, name, threshold=None):
        """
        Returns a pandas DataFrame for the specified CSV file.
        If the DataFrame is already loaded, it returns the cached version.
        """
        assert isinstance(name, str) or isinstance(name, pd.DataFrame), f"Expected str, got {type(name)}"
        
        # if name in self.dataframes:
        #     return self.dataframes[name]

        if (self.multiclass or name[:6]=='params'):
            df = self._pd(name)
        elif not self.multiclass:
            assert threshold is not None, "Threshold must be specified for single class runs."
            df = self.multi_index(name)
            df = df.xs(threshold, level='Thresholds').reset_index()
        
        self.dataframes[name] = df
        return df
        

        
    
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
    def get_params_header(self):
        metrics_path = self.run.results_folder / "params_train.csv"
        df = pd.read_csv(metrics_path, nrows=0)  
        columns = df.columns.tolist()
        return [col for col in columns if col != "Epoch"]
        

    #################################### Plot helpers


    def _plot(self, df, title=' ', xlabel="Epoch", figsize=(10, 6), epoch=-1, log=False, save=False):
        """
        Plots the DataFrame with specified title and labels.
        If save is True, saves the plot to the outputs folder.
        """
        # Elenco fisso di colori da usare ciclicamente
        fixed_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        if xlabel == 'Thresholds':
            exclude = 'Thresholds'
            epoch = -1
        else:
            exclude = 'Epoch'
        if epoch != -1:
            df = df[df['Epoch'] <= epoch]
        plt.figure(figsize=figsize)
        color_index = 0
        for col in df.columns:
            if col != exclude:
                color = fixed_colors[color_index % len(fixed_colors)]
                plt.plot(df[xlabel], df[col], label=col, color=color)
                color_index += 1

        plt.xlabel(xlabel)
        if log:
            plt.yscale('log')
        # insert a grid
        plt.grid(True)
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



    def plot(self, name, columns=None, scale=None, threshold=None, figsize=(10, 6), epoch=-1, log=False, save=False):
        """
        Plots specified columns (or all columns if None) against 'Epoch'
        from a CSV in the run's results folder.
        """
        # if isinstance(name, str):
        #     if self.multiclass:
        #         df = self.pd(name)
        #     if not self.multiclass:
        #         assert threshold is not None, "Threshold must be specified for single class runs."
        #         df = self.multi_index(name)
        #         df = df.xs(threshold, level='Thresholds').reset_index()

        if isinstance(name, str):
            df = self.pd(name, threshold=threshold)
        elif isinstance(name, pd.DataFrame):
            df = name
            name = df.name if hasattr(df, 'name') else " "
        else:
            raise TypeError(f"Expected str or pd.DataFrame, got {type(name)}")
        

        if df.empty:
            raise ValueError(f"CSV {name} is empty.")
        
        plot_columns = columns or [col for col in df.columns if col != 'Epoch']

        _df = df[['Epoch'] + plot_columns]

        if scale is not None:
            assert len(scale) == len(columns), "Scale list must match number of columns."
            for i, sc in enumerate(scale):
                _df[columns[i]] *= sc

        self._plot(df=_df, title=name, xlabel="Epoch", figsize=figsize, epoch=epoch, log=log, save=save)

    def which_csv(self, column):
        """
        Returns the setting (metrics or IoU) based on the column name.
        """
        if column in self.metrics_header:
            return 'metrics'
        elif column in self.run.Dataset.class_names:
            return 'IoU'
        elif column in self.common_metrics_header:
            return 'params'
        else:
            raise ValueError(f"Column '{column}' is not a valid metric or class name.")


    def plot_train_val(self, column, epoch=-1, save=False, threshold=None, log=False, diff=False):
        """
        Plots the specified column for both training and validation datasets.
        """
        
        train_df = self.pd(f'{self.which_csv(column)}_train', threshold=threshold)
        val_df = self.pd(f'{self.which_csv(column)}_val', threshold=threshold)

        if train_df.empty or val_df.empty:
            raise ValueError("Train or validation DataFrame is empty.")

        combined = self.combine({'train': train_df, 'val': val_df}, column)

        if diff:
            combined['Diff'] = combined['train'] - combined['val']

        self._plot(combined, title=f"{column}", xlabel="Epoch", epoch=epoch, log=log, save=save)
        return combined

    def plot_runs(self, runs, column, set_, epoch=50, diff=False, save=False, log=False, threshold=None):  
        """
        Plots the specified column for multiple runs.
        """
        if isinstance(runs, str):
            runs = [runs]
        
        dfs = {}
        for run in runs:
            plotter = Plotter(run)
            if diff:
                df_train = plotter.pd(f'{plotter.which_csv(column)}_train', threshold=threshold)
                df_val = plotter.pd(f'{plotter.which_csv(column)}_val', threshold=threshold)
                df = pd.DataFrame()
                df['Epoch'] = df_train['Epoch']
                df[column] = df_train[column] - df_val[column]
            else:
                df = plotter.pd(f'{self.which_csv(column)}_{set_}', threshold=threshold)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in run {run}.")
            dfs[run] = df[['Epoch', column]]

        combined = self.combine(dfs, column)

        self._plot(combined, title=f"{column} across runs", xlabel="Epoch", epoch=epoch, log=log, save=save)
        return combined
    @property
    def sorted_classes(self):
        """
        Returns the class names sorted by pixel count in ascending order.
        """

        class_names, pixel_count = self.run.Dataset.class_names, self.run.Dataset.pixel_count

        l = dict(zip(class_names, pixel_count))
        l = dict(sorted(l.items(), key=lambda item: item[1]))

        return list(reversed(l.keys()))

    def plot_sorted_IoU(self, epoch=-1,save=False, diff=False):
        l = self.sorted_classes
        for label in l:
            self.plot_train_val(label, epoch=epoch, save=save, diff=diff)



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

        
    ##################################### Threshold Split

    def is_list_col(self, df, col):
        return (
            df[col].astype(str).str.contains(":").all() and
            df[col].str.count(":").eq(df["Thresholds"].str.count(":")).all()
        )
    
    def multi_index(self, name):
        if self.multiclass:
            raise ValueError("MultiIndex is only supported for single class runs, which have thresholds.")
        df = self._pd(name).copy(deep=True)

        # discriminate columns
        cols = df.columns.tolist()
        base_cols = ["Epoch", "Time", "Loss"]
        list_cols = [col for col in cols if col not in base_cols]

        # check for colums whose values are expected to be lists
        for col in list_cols:
            if not self.is_list_col(df, col):
                raise ValueError(f"Column {col} is not a list column.")
        
        # converts columns elements to actual lists of floats
        listize = lambda x: list(map(float, x.split(":")))
        for col in list_cols:
            df[col] = df[col].apply(listize)

        # transforms every element in a list of the same length as the "Thresholds" list
        for col in base_cols:
            df[col] = df.apply(lambda row: [row[col]] * len(row["Thresholds"]), axis=1)


        # Esplodi tutto
        df_exp = df.explode(list(df.columns))

        # Imposta MultiIndex
        df_exp["Epoch"] = df_exp["Epoch"].astype(int)
        df_exp.set_index(["Epoch", "Thresholds"], inplace=True)
        df_exp.sort_index(inplace=True)

        return df_exp
    

    def plot_threshold(self, name, epoch=-1, columns=None, save=False):
        assert columns is not None, "Columns must be specified for threshold plots."
        df = self.multi_index(name)
        if epoch == -1:
            epoch = df.index.get_level_values('Epoch').max()
        df = df.xs(epoch, level='Epoch').reset_index()
        plot_columns = columns or [col for col in df.columns if col != 'Thresholds']
        plt.figure(figsize=(10, 6))
        
        for col in plot_columns:
            plt.plot(df["Thresholds"], df[col], label=col)
            plt.scatter(df["Thresholds"], df[col], s=50)  # Pallini


        plt.xlabel("Thresholds")
        plt.ylabel("Metric Value")
        plt.title(f"Threshold Analysis - Epoch {epoch}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            save_path = self.run.outputs_folder / f"threshold_plot_{name}_epoch_{epoch}.png"
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")

        plt.show()


    def binary_cm(self, name, epoch, threshold):

        df = self.multi_index(name)
        if epoch == -1:
            epoch = df.index.get_level_values('Epoch').max()

        df = df.loc[(epoch, threshold)]
      
        cm = [[df['TN']/(df['TN'] + df['FP']), df['FP']/(df['TN'] + df['FP'])],
                [df['FN']/(df['FN'] + df['TP']), df['TP']/(df['FN'] + df['TP'])]]
        conf_matrix = pd.DataFrame(cm,
            index=pd.Index(['Actual Negative', 'Actual Positive']),
            columns=pd.Index(['Predicted Negative', 'Predicted Positive'])
        )

        # Plot con matplotlib
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Annotazioni sulle celle
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, f'{conf_matrix.iloc[i, j]:,.3f}',
                        ha='center', va='center', color='black')

        plt.xticks(range(2), conf_matrix.columns)
        plt.yticks(range(2), conf_matrix.index)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        