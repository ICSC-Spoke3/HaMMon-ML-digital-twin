import torch
from torchmetrics.classification import BinaryConfusionMatrix

class BynaryMetrics:
    """
    A wrapper around multiple BinaryConfusionMatrix instances, one per threshold.
    Provides: to(device), update(preds, targets), compute(), reset().
    """

    def __init__(self, thresholds):
        """
        thresholds: float or list of floats (e.g. 0.5 or [0.1, 0.2, ..., 0.9])
        """
        if isinstance(thresholds, float):
            self.thresholds = [thresholds]
        else:
            self.thresholds = thresholds
        
        # Create a confusion matrix object for each threshold
        self.cm_list = [BinaryConfusionMatrix() for _ in self.thresholds]

    def to(self, device):
        """
        Moves all confusion matrix objects to a specific device.
        """
        for cm in self.cm_list:
            cm.to(device)
        return self  # so we can do chaining


    def update(self, preds: torch.Tensor, target: torch.Tensor, detach_inputs=True):
        """
        Updates metrics for the given predictions and targets.

        Args:
            preds (torch.Tensor): Predictions (logits or probabilities).
            target (torch.Tensor): Ground truth (binary 0/1).
            detach_inputs (bool): If True (default), detaches inputs to avoid gradient tracking.
        """
        if detach_inputs:
            _preds = preds.detach()
            _target = target.detach()

        for i, thr in enumerate(self.thresholds):
            bin_preds = (_preds >= thr).long()
            self.cm_list[i].update(bin_preds, _target)



    def compute(self):
        """
        Returns a dictionary where each key is a metric and the value is a list of results for each threshold:
        {
            "threshold": [0.1, 0.2, ..., 0.9],
            "accuracy": [...],
            "precision": [...],
            "recall": [...],
            "dice_score": [...],
            "iou": [...],
            "confusion_matrix": [...]
        }
        """
        results = {
            "threshold": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "dice": [],
            "iou": [],
            "confusion_matrix": []
        }

        for thr, cm in zip(self.thresholds, self.cm_list):
            c_matrix = cm.compute()  # Compute confusion matrix
            c_matrix_np = c_matrix.cpu().numpy() if c_matrix.is_cuda else c_matrix.numpy()
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
            results["threshold"].append(thr)
            results["confusion_matrix"].append([[TN, FP], [FN, TP]])
            results["accuracy"].append(accuracy)
            results["precision"].append(precision)
            results["recall"].append(recall)
            results["dice"].append(dice_score)
            results["iou"].append(iou)

        return results


    def reset(self):
        """
        Resets all confusion matrix objects.
        """
        for cm in self.cm_list:
            cm.reset()
