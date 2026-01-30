# Project Structure Overview

This file summarizes the main components of the repository. The code lives under `src/` and the datasets are stored in the `datasets/` directory.

## Runner

The training loop is implemented in `src/runner.py`. The `Runner` class inherits from `SetObjects` and `SetMetrics` (defined in `runner_set_objects.py` and `runner_set_metrics.py`) and orchestrates the training epochs. Key responsibilities include:

- **`SetObjects`** – loads the datasets, builds the DataLoaders, initializes the model and attaches the optimizer, scheduler and criterion, restoring checkpoints when needed.
- **`SetMetrics`** – configures the metrics to compute for each subset (train, val, test) based on the run configuration and prepares the CSV logs.
- `epoch_train()` and `epoch_eval()` perform a single epoch of training and validation respectively.
- `loop()` iterates through epochs, saving checkpoints and logging metrics through the `Run` object.

## Run Object

The `Run` class (`src/run.py`) manages configuration, file paths and saving/loading of objects. It reads `config.yaml` and tracks its history, creates folders for results, outputs and data, and provides methods to:

- save checkpoints (weights, optimizer, scheduler) per epoch
- log results to CSV files
- retrieve the last saved epoch to resume training
- clear results or data when needed

During initialization it also dynamically imports the dataset module specified in the configuration.

## Dataset Objects

Datasets are defined in the `datasets/` directory (for example `floodnet_resized.py` and `kaggle_crack.py`). Each dataset exposes a `Dataset` class with a common API. The class must provide at least the following attributes so that the rest of the code can access dataset statistics:

- `class_names`: list of class labels
- `class_colors`: list of RGB tuples matching the class order
- `mean` and `std`: raw RGB statistics used for normalization (and derived `norm_mean`, `norm_std`)
- `image_count`: optional count of images per class
- `pixel_count`: number of pixels per class

## Libraries in `src/`

- **`binary_metrics.py`** – helpers to compute binary segmentation metrics at multiple thresholds.
- **`csv_logger.py`** – small utility to append metrics to CSV files with basic validation.
- **`imgs.py`** – visualize dataset images, labels and overlays.
- **`plotter.py`** – load CSV logs and plot metrics, IoU trends and confusion matrices.
- **`runner.py`** – main training loop, relies on `SetObjects` and `SetMetrics`.
- **`runner_set_metrics.py`** – initializes accuracy, IoU and confusion matrix metrics and creates CSV files for train/val/test results.
- **`runner_set_objects.py`** – attaches dataset loaders, models, optimizer, scheduler, loss function and optional patcher to the `Runner`.
- **`tools.py`** – simple memory usage printer used during debugging.
- **`tyaml.py`** – wrapper to track changes in YAML configuration files.
- **`settings.py`** – reads global `settings.yaml` containing paths and distributed training options.
- **`run.py`** – definition of the `Run` helper used by training scripts.
- **`utils.py`** – functions for computing class weights and normalization utilities.

These modules work together to build training scripts for semantic segmentation experiments.
