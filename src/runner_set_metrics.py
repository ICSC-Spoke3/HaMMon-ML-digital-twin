import logging


class SetMetrics:
    def __init__(self, run, rank):
        self.run = run
        self.rank = rank

        if 'multiclass' not in self.run.config:
            self.run.config['multiclass'] = True  # Default to True if not set
        
        if self.run.config['multiclass'] == True:
            from src.metrics_multiclass import Metrics
        elif self.run.config['multiclass'] == False:
            from src.metrics_binary import BinaryMetrics as Metrics
        else:
            raise ValueError("Invalid value for 'multiclass' in run.config. Expected True or False.")

 
        # if run.config.metrics does not exist, create it with default value
        if 'metrics' not in self.run.config:
            self.run.config['metrics'] = {
                "train": ['accuracy','iou'],
                "val": ['accuracy','iou','cm'],
            }
        if 'train' not in self.run.config['metrics']:
            raise ValueError("Metrics for 'train' subset must be defined in run.config['metrics']")
        if 'val' not in self.run.config['metrics']:
            raise ValueError("Metrics for 'val' subset must be defined in run.config['metrics']")

        self.metrics = {}
        for subset in self.run.config['metrics']:
            self.metrics[subset] = Metrics(
                run=self.run,
                rank=self.rank,
                subset=subset,
                config=self.run.config['metrics'][subset]
            )
        