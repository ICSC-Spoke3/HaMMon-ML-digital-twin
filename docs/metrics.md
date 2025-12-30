## Metrics and plots

This repo does not rely on third-party tools for tracking training metrics or
parameters. Everything is logged to CSV via `src/csv_logger.py`.

Metrics are collected under the `results` folder of each training run, split
across multiple CSV files. They can be visualized with pandas; lightweight
wrappers and plotting utilities live in `src/plotter.py`.

Tracked training metrics include time, loss/error, confusion matrix (CM), and
IoU per class, plus common classification scores (accuracy, precision, recall,
dice).

For single-class training, values are reported for a set of threshold values
that can be configured in the config file, enabling threshold analysis.
