 
import logging

class Metrics:
    def __init__(self, runner, subset):
        assert isinstance(subset, str), f"subset must be a string, but got {type(subset)}"

        self.runner = runner
        self.run = self.runner.run
        self.rank = self.runner.rank
        self.subset = subset

        self.update_count  = 0

        if "params" not in self.run.config:
            self.params = []
            self.metrics = {}
        else:
            if not isinstance(self.run.config["params"], list):
                raise ValueError(f"Expected 'params' in config to be a list, but got {type(self.run.config['params'])}")
            self.params = self.run.config["params"]

            self.metrics = {param: 0.0 for param in self.params}

        if self.rank == 0:
            if len(self.params) > 0:
                self.run.new_csv(f'params_{subset}', header=self.params)
            else:
                raise ValueError(f"No parameters defined for {subset} in config.yaml")

    def reset(self):
        for metric in self.metrics:
            self.metrics[metric] = 0.0
        self.update_count = 0

    def update(self, output, target):
        pred = output.argmax(dim=1)  # Get the predicted class indices
        for metric in self.metrics:
            match metric:
                case 'lr':
                    if not self.runner.eval:
                        logging.debug(f'update {metric}: {self.metrics[metric]}')
                case 'loss_focal':
                    self.metrics[metric] += self.runner.criterion.last_focal.item()
                    logging.debug(f'update {metric}: {self.metrics[metric]}')
                case 'loss_dice':
                    self.metrics[metric] += self.runner.criterion.last_dice.item()
                    logging.debug(f'update {metric}: {self.metrics[metric]}')
                case 'loss':
                    self.metrics[metric] += self.runner.criterion.last_loss.item()
                    logging.debug(f'update {metric}: {self.metrics[metric]}')
        self.update_count += 1
        logging.debug(f"Update count: {self.update_count}")


    def compute(self):
        self.results = {}
        for metric in self.metrics:
            match metric:
                case 'lr':
                    if not self.runner.eval:
                        self.metrics[metric] = self.runner.optimizer.param_groups[0]['lr']  # Get the last learning rate
                        logging.debug(f"lr: {self.metrics[metric]}")
                case 'loss_focal':
                    self.metrics[metric] = self.metrics[metric] / self.update_count
                    logging.debug(f"Focal loss: {self.metrics[metric]}")
                case 'loss_dice':
                    self.metrics[metric] = self.metrics[metric] / self.update_count
                    logging.debug(f"Dice loss: {self.metrics[metric]}")
                case 'loss':
                    self.metrics[metric] = self.metrics[metric] / self.update_count
                    logging.debug(f"Loss: {self.metrics[metric]}")
        return self.results

    def save(self, epoch, elapsed, loss):
        if self.rank == 0:
            # Log metrics to CSV
            self.run.log_csv(f'params_{self.subset}',  epoch, self.metrics)
