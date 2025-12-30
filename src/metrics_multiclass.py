import torch
from torchmetrics.classification import MulticlassConfusionMatrix 


import logging
logging.warning('pulire metrics_multiclass.py')


class Metrics:
    def __init__(self, runner, subset):
        assert isinstance(subset, str), f"subset must be a string, but got {type(subset)}"

        self.runner = runner
        self.run = self.runner.run
        self.rank = self.runner.rank
        self.subset = subset

        self.metrics = {}

        if self.rank == 0:

            # metrics_{subset}.csv
            results_header = ["Time", "Loss", "Thresholds", "Accuracy", "Precision", "Recall", "Dice", "Error", "IoU"]
            self.run.new_csv(f'metrics_{subset}', header=results_header)

            # IoU_{subset}.csv
            iou_header = self.run.Dataset.class_names
            self.run.new_csv(f'IoU_{subset}', header=iou_header)

            # CM_{subset}.csv
            CM_header = [str(i) for i in range(1, len(self.run.Dataset.class_names)**2 + 1)]
            self.run.new_csv(f'CM_{subset}', header=CM_header)

        self.cm = MulticlassConfusionMatrix(
            num_classes=len(self.run.Dataset.class_names),
            sync_on_compute=True,
            ignore_index=255
        ).to(self.rank)

    
    def reset(self):
        self.cm.reset()

    def update(self, output, target):
        pred = output.argmax(dim=1)  # Get the predicted class indices
        self.cm.update(pred.view(-1), target.view(-1))
                   

    def compute(self):
        self.results = {}
        self.results['cm'] = self.cm.compute()
        return self.results

    def save(self, epoch, elapsed, loss):
        if self.rank == 0:
            # Log metrics to CSV

            m = from_cm(self.results['cm'])

            self.run.log_csv(f'metrics_{self.subset}', epoch, 
                                {'Time': elapsed, 
                                'Loss': loss,
                                'Accuracy': m['accuracy'].item(),
                                'Precision': m['precision'].item(),
                                'Recall': m['recall'].item(),
                                'Dice': m['dice'].item(),
                                'IoU': m['iou'].item()})
            
            self.run.log_csv(f'IoU_{self.subset}', epoch, m['iou_per_class'].cpu().numpy().tolist())
            self.run.log_csv(f'CM_{self.subset}', epoch, self.results['cm'].cpu().numpy().flatten().tolist())



def from_cm(cm_tensor):
    """ Convert a confusion matrix tensor to a dictionary of metrics."""
    assert isinstance(cm_tensor, torch.Tensor), "Input must be a torch.Tensor"

    # per class lists
    TP = cm_tensor.diag()                       
    FP = cm_tensor.sum(dim=0) - TP
    FN = cm_tensor.sum(dim=1) - TP
    TN = cm_tensor.sum() - (TP + FP + FN)


    # avoid division by zero
    iou_per_class = TP / (TP + FP + FN).clamp(min=1e-6)
    accuracy = TP.sum() / cm_tensor.sum()
    precision = TP / (TP + FP).clamp(min=1e-6)
    recall = TP / (TP + FN).clamp(min=1e-6)
    dice = (2 * TP) / (2 * TP + FP + FN).clamp(min=1e-6)
    weights = TP + FN

    def weighted_mean(values, weights):
        """Calculate the weighted mean of values."""
        return (values * weights).sum() / weights.sum().clamp(min=1e-6)

    return {
        "weights": weights,
        "iou_per_class": iou_per_class,
        "accuracy_per_class": accuracy,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "dice_per_class": dice,
        "iou": iou_per_class.mean(),
        "accuracy": accuracy.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "dice": dice.mean()
    }