import torch
from torchmetrics.classification import BinaryConfusionMatrix

import logging

DECIMALS = 5 # Number of decimal places for metrics output

from src.predict import Predict



class BinaryMetrics:
    """
    A wrapper around multiple BinaryConfusionMatrix instances, one per threshold.
    Provides: to(device), update(preds, targets), compute(), reset().
    """
    def __init__(self, runner, subset):
        assert isinstance(subset, str), f"subset must be a string, but got {type(subset)}"


        self.runner = runner
        self.run = self.runner.run
        self.rank = self.runner.rank
        self.subset = subset

        self.metrics = {}

        if 'thresholds' not in self.run.config or self.run.config["thresholds"] is None:
            self.thresholds = [i/10 for i in range(1, 10)]  # Default thresholds from 0.1 to 0.9
        else:
            self.thresholds = self.run.config["thresholds"]

 
        if self.rank == 0:
            results_header = ["Time", "Loss", "Thresholds", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Dice", "IoU"]
            self.run.new_csv(f'metrics_{subset}', header=results_header)
        self.cm_results = [] # Store results for each threshold
        # Initialize a BinaryConfusionMatrix for each threshold
        self.cms = [BinaryConfusionMatrix(sync_on_compute=True).to(self.rank) for _ in self.thresholds]

        self.binary_predictions = Predict('binary_predictions')



    def reset(self):
        for cm in self.cms:
            cm.reset()
        self.cm_results= []

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates metrics for the given predictions and targets.

        Args:
            preds (torch.Tensor): Predictions (logits or probabilities).
            target (torch.Tensor): Ground truth (binary 0/1).
        """

        for i, thr in enumerate(self.thresholds):
            bin_preds =self.binary_predictions(preds, threshold=thr)
            self.cms[i].update(bin_preds, target.unsqueeze(1))
                  
    def compute(self):
        for i in range(len(self.thresholds)):
            self.cm_results.append(self.cms[i].compute())
 
    def save(self, epoch, elapsed, loss):
        """
        Saves the computed metrics to a CSV file.

        Args:
            epoch (int): Current epoch number.
            elapsed (float): Time elapsed for the epoch.
            loss (float): Loss value for the epoch.
        """

        results_header = ["Time", "Loss", "Thresholds", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "Dice", "IoU"]

        results = {
            "Thresholds": [],
            "TN": [],
            "FP": [],
            "FN": [],
            "TP": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "Dice": [],
            "IoU": []
        }

        for thr, cm in zip(self.thresholds, self.cm_results):
            c_matrix_np = cm.cpu().numpy() if cm.is_cuda else cm.numpy()
            TN, FP, FN, TP = c_matrix_np.ravel()

            # Calculate metrics
            denom_all = TN + FP + FN + TP
            accuracy = (TN + TP) / denom_all if denom_all > 0 else 0.0

            denom_prec = TP + FP
            precision = TP / denom_prec if denom_prec > 0 else 0.0

            denom_rec = TP + FN
            recall = TP / denom_rec if denom_rec > 0 else 0.0

            denom_dice = 2 * TP + FP + FN
            dice_score = 2 * TP / denom_dice if denom_dice > 0 else 0.0

            denom_iou = TP + FP + FN
            iou = TP / denom_iou if denom_iou > 0 else 0.0

            # Append results to the dictionary
            results["Thresholds"].append(thr)
            results["TN"].append(TN)
            results["FP"].append(FP)
            results["FN"].append(FN)
            results["TP"].append(TP)
            results["Accuracy"].append(accuracy)
            results["Precision"].append(precision)
            results["Recall"].append(recall)
            results["Dice"].append(dice_score)
            results["IoU"].append(iou)

        results_str = {
            key: ":".join(str(round(val, DECIMALS)) for val in values)
            for key, values in results.items()
        }
        results_str["Time"] = elapsed
        results_str["Loss"] = loss

        self.run.log_csv(f'metrics_{self.subset}', epoch, results_str)
