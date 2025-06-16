 
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassConfusionMatrix 
from torchmetrics.classification import MulticlassAccuracy


class Metrics:
    def __init__(self, run, rank, subset, config):
        assert isinstance(subset, str), f"subset must be a string, but got {type(subset)}"

        self.run = run
        self.rank = rank
        self.subset = subset

        self.config = config

        implemented = ['accuracy', 'iou', 'cm']
        # control if metrics values inside config are unique
        if len(self.config) != len(set(self.config)):
            raise ValueError(f"Metrics for {subset} must be unique, but found duplicates in {self.config}")
        
        # control if metrics values inside config are implemented
        for metric in self.config:
            if metric not in implemented:
                raise NotImplementedError(f"Metric {metric} is not implemented.")

        self.metrics = {}


        for metric in self.config:
            match metric:

                case 'accuracy':
                    if self.rank == 0:
                        results_header = ["Time", "Loss", "Thresholds", "Accuracy", "Precision", "Recall", "Dice", "Error", "IoU"]
                        self.run.new_csv(f'metrics_{subset}', header=results_header)

                    self.metrics[metric] = MulticlassAccuracy(
                        num_classes=len(self.run.Dataset.class_names),
                        average='micro',
                        sync_on_compute=True
                    ).to(self.rank)

                case 'iou':
                    if self.rank == 0:
                        # Initialize CSV file for IoU results
                        iou_header = self.run.Dataset.class_names
                        self.run.new_csv(f'IoU_{subset}', header=iou_header)
                    self.metrics[metric] = JaccardIndex(
                        task='multiclass', 
                        num_classes=len(self.run.Dataset.class_names), 
                        average="none",
                        sync_on_compute=True
                    ).to(self.rank)

                case 'cm':
                    if self.rank == 0:
                        # Initialize CSV file for confusion matrix results
                        CM_header = [str(i) for i in range(1, len(self.run.Dataset.class_names)**2 + 1)]
                        self.run.new_csv(f'CM_{subset}', header=CM_header)
                    self.metrics[metric] = MulticlassConfusionMatrix(
                        num_classes=len(self.run.Dataset.class_names),
                        sync_on_compute=True
                    ).to(self.rank)
                case _:
                    raise ValueError(f"Metric {metric} is not implemented.")
    
    def reset(self):
        for metric in self.metrics:
            self.metrics[metric].reset()
    
    def update(self, pred, target):
        for metric in self.metrics:
            match metric:
                case 'accuracy':
                    self.metrics[metric].update(pred.view(-1), target.view(-1))
                case 'iou':
                    self.metrics[metric].update(pred, target)
                case 'cm':
                    self.metrics[metric].update(pred.view(-1), target.view(-1))

    def compute(self):
        self.results = {}
        for metric in self.metrics:
            match metric:
                case 'accuracy':
                    self.results['accuracy'] = self.metrics[metric].compute()
                case 'iou':
                    self.results['iou'] = self.metrics[metric].compute()
                case 'cm':
                    self.results['cm'] = self.metrics[metric].compute()
        return self.results

    def save(self, epoch, elapsed, loss):
        if self.rank == 0:
            # Log metrics to CSV

            for metric in self.metrics:
                if metric == 'accuracy':
                    acc = self.results['accuracy']
                    
                    iou = self.results['iou']
                    self.run.log_csv(f'metrics_{self.subset}', epoch, {'Time': elapsed, 'Loss': loss,'Accuracy': acc.item(),'IoU': iou.mean().item()})
                  
                elif metric == 'iou':
                    self.run.log_csv(f'IoU_{self.subset}', epoch, self.results['iou'].cpu().numpy().tolist())
                elif metric == 'cm':
                    self.run.log_csv(f'CM_{self.subset}', epoch, self.results['cm'].cpu().numpy().flatten().tolist())
