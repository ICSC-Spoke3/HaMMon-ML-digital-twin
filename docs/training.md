# Training Instructions

This document provides instructions to set up and launch a distributed training run using this repository.

Follow these guidelines to ensure reproducibility and smooth execution.

# Environment Configuration

Before starting a new training run, copy `settings_example.yaml` to `settings.yaml` in the repository root. Adjust the paths to match your system and set the distributed training parameters:

* `run_folder` and `data_folder`: destination directories for run outputs and checkpoint data.
* `datasets_folder`: location of the datasets.
* `backend`, `MASTER_ADDR`, `MASTER_PORT`: communication settings for PyTorch's distributed backend.
* `world_size`: number of GPUs (processes) used for training.
* `num_workers`: DataLoader workers per process.

# Creating a Run Folder

Each training run resides in a folder under the path specified by `run_folder`. Create a new directory inside `.runs/` (or the custom run folder you set in `settings.yaml`).

Inside this directory place:

* `runner_init.py` – a Python script that defines one or more classes derived from `Runner` (for example `TrainRunner` and `TestRunner`).
  Each class implements methods such as `set_dataset_train`, `set_dataset_eval`, `set_model`, `set_optimizer`,
  `set_scheduler` and `set_criterion` in order to build the training or validation pipeline.
* `config.yaml` – a YAML file containing hyperparameters such as dataset name, batch size and learning rate. The `Run` class reads this file when training starts and tracks any configuration changes by storing file hashes in `config-history.yaml`.

Use one of the existing folders inside `.runs` as a template if needed.

# Runner loop

`runner.py` defines the training/validation loop for a single run. It consumes `run.config`, builds the objects provided by `runner_init.py`, and orchestrates the per‑epoch flow (reset/update/compute/save metrics, checkpointing, and evaluation). The `Run` class in `run.py` owns the run context: it resolves folders, loads `config.yaml`, tracks config changes via `Tyaml`, and provides the CSV logging and checkpoint I/O used by the runner.


# Launching Training

Once `settings.yaml` and the run folder are ready, start training with:

```bash
python ddp.py <run-name>
```

`<run-name>` must match the folder name you created under `run_folder`. The script loads `runner_init.py` from that folder, reads the configuration, and launches distributed training using the parameters from `settings.yaml`.

# Launching Evaluation on Test Set

To run evaluation on the test split, instantiate `TestRunner` in `runner_init.py` and make sure `config.yaml` includes the evaluation-specific settings (for example, dataset split, batch size, and any flags required by the test pipeline). The `Run` class will load the updated configuration and use the `TestRunner` hooks when launching the evaluation.

```bash
python ddp.py <run-name> --test <test-name>
```

# Clear Flag

Use `--clear` to remove the existing run folder contents before starting. This is helpful when you want a clean slate for logs, checkpoints, and cached artifacts without manually deleting the directory.
