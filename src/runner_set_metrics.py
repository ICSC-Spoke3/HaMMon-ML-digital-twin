import logging
from src.metrics_common import Metrics as CommonMetrics


class Metrics:
    def __init__(self, runner, subset):
        assert isinstance(subset, str), f"subset must be a string, but got {type(subset)}"

        self.runner = runner
        self.subset = subset
        self.run = self.runner.run
        self.rank = self.runner.rank

        if self.run.config['multiclass'] == True:
            from src.metrics_multiclass import Metrics as _Metrics
        elif self.run.config['multiclass'] == False:
            from src.metrics_binary import BinaryMetrics as _Metrics
        else:
            raise ValueError("Invalid value for 'multiclass' in run.config. Expected True or False.")
        
        self.metrics = _Metrics(
            runner=self.runner,
            subset=self.subset,
        )
        
        self.common_metrics = CommonMetrics(
            runner=self.runner,
            subset=self.subset,
        )
    

    def reset(self):
        self.metrics.reset()
        self.common_metrics.reset()

    def update(self, output, target):
        self.metrics.update(output, target)
        self.common_metrics.update(output, target)
    
    def compute(self):
        self.metrics.compute()
        self.common_metrics.compute()
    
    def save(self, epoch, elapsed, loss):
        self.metrics.save(epoch, elapsed, loss)
        self.common_metrics.save(epoch, elapsed, loss)

        

    

class SetMetrics:
    def __init__(self, run, rank):
        self.run = run
        self.rank = rank

        if 'multiclass' not in self.run.config:
            self.run.config['multiclass'] = True  # Default to True if not set
 
        self.metrics = {}
 
    
    def set_metrics(self, subset):
        assert isinstance(subset, str), f"subset must be a string, but got {type(subset)}"
        self.metrics[subset] = Metrics(
            runner=self,
            subset=subset,
        )

        