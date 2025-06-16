import time
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
DEBUG=logging.getLogger().isEnabledFor(logging.DEBUG)

import torch

import sys

from src.runner_set_objects import SetObjects
from src.runner_set_metrics import SetMetrics
from src.tools import Tools
from src.run import Run


# Set to FALSE during actual runs
EARLY_BREAK = False


class Runner(SetMetrics, SetObjects):
    def __init__(self, run: Run, rank):
        assert isinstance(run, Run), "run must be an instance of Run class"
        assert isinstance(rank, int), "rank must be an integer"
        self.run = run
        self.rank = rank
        self.t = Tools(DEBUG=DEBUG)
      

        SetMetrics.__init__(self, run, rank)
        SetObjects.__init__(self, run, rank)
    
    @property
    def lew(self):
        return self.run.get_lew()


    def epoch_train(self, epoch):
        assert isinstance(epoch, int), "epoch must be an integer"

        since = time.time()

        self.metrics['train'].reset()  # Reset metrics for the training epoch
  
        trn_loss = 0

        self.dataloader_train.sampler.set_epoch(epoch)

        for i, data in enumerate(self.dataloader_train): 

            if EARLY_BREAK is not None and i >= EARLY_BREAK:
                    logging.warning(f"rank {self.rank}: Training Early break at batch {i}")
                    break
    
            

            self.optimizer.zero_grad()

            inputs = data[0].to(self.rank)
            targets = data[1].to(self.rank)

            self.t.memprint(f"batch {i} inputs {inputs.shape}, targets {targets.shape}")

            output = self.model(inputs)
            self.t.memprint(f' outputs {output.shape}, targets {targets.shape}')

            loss = self.criterion(output, targets)
            trn_loss += loss.item() # out of the computation graph
            self.t.memprint(f'loss: {loss.item()}, trn_loss: {trn_loss}')

            loss.backward()
            self.t.memprint('after backprop')

            self.optimizer.step()
            self.t.memprint('after step')

            pred = output.argmax(dim=1) # max index along the channel dimension

            self.metrics['train'].update(pred, targets)  # Update training metrics

        trn_loss /= len(self.dataloader_train)
        self.scheduler.step(trn_loss)

        self.metrics['train'].compute()
      

        elapsed = time.time() - since

        return elapsed, trn_loss



    def epoch_eval(self, epoch, idx):
        assert isinstance(epoch, int), "epoch must be an integer"
        assert isinstance(idx, str), "idx must be a string"

        since = time.time()
        self.model.eval()

        self.metrics[idx].reset()  # Reset metrics for the validation epoch
       

        val_loss = 0

        
        self.dataloader_eval.sampler.set_epoch(epoch)

        with torch.no_grad():       
            for i, data in enumerate(self.dataloader_eval):

                if EARLY_BREAK is not None and i >= EARLY_BREAK:
                        logging.warning(f"rank {self.rank}: Eval Early break at batch {i}")
                        break
                

                inputs = data[0].to(self.rank)
                targets = data[1].to(self.rank)

                self.t.memprint(f"rank {self.rank} batch {i} inputs {inputs.shape}, targets {targets.shape}")

                if self.patcher is not None:
                    output = self.patcher(inputs)
                else:
                    output = self.model(inputs)

                self.t.memprint(f"rank {self.rank} outputs {output.shape}, targets {targets.shape}")

                loss = self.criterion(output, targets)
                val_loss += loss.item()            

                pred = output.argmax(dim=1)

                self.metrics[idx].update(pred, targets)
      

        val_loss /= len(self.dataloader_eval)

        self.metrics[idx].compute()# Update validation metrics

        elapsed = time.time() - since

        return elapsed, val_loss 


    def loop(self):

        for epoch in range(self.lew + 1, self.run.config["epochs"] + 1):

            logging.info(f'rank {self.rank}: Starting Training Epoch: {epoch}')

            # tracking changes in the config file if rank = 0
            if self.rank == 0:
                self.run.track(epoch)

            #---------------------------------------------- training

            elapsed, loss = self.epoch_train(epoch)

            if self.rank == 0:
                self.metrics['train'].save(epoch, elapsed, loss)  # Save training metrics
               
                self.run.save_weights(self.model.module.state_dict(), epoch)
                self.run.save('optimizer', self.optimizer.state_dict(), epoch)
                self.run.save('scheduler', self.scheduler.state_dict(), epoch)

            #---------------------------------------------- validation

            elapsedv, lossv = self.epoch_eval(epoch, 'val')

            if self.rank == 0:
                self.metrics['val'].save(epoch, elapsedv, lossv)

            if EARLY_BREAK is not None and epoch >= EARLY_BREAK:
                    logging.info(f"rank {self.rank}: Early break at epoch {epoch}")
                    break

    def loop_eval(self, idx='test'):
        assert isinstance(idx, str), "idx must be a string"

        epochs = self.run.config["eval_epochs"]
        epochs = tuple(map(int, epochs.split(',')))
        assert len(epochs) == 2, "eval_epochs must be a tuple of two integers (start, end)"

        if idx not in self.metrics:
            raise ValueError(f"Test set {idx} missing in config.yaml")


        for epoch in range(epochs[0], epochs[1] + 1):

            self._set_model(epoch=epoch)

            logging.info(f'rank {self.rank}: Starting {idx} Epoch: {epoch}')

            elapsedv, lossv = self.epoch_eval(epoch, idx)

            if self.rank == 0:
                self.metrics[idx].save(epoch, elapsedv, lossv)



    