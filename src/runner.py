import time
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
DEBUG=logging.getLogger().isEnabledFor(logging.DEBUG)

import torch
import torch.distributed as dist



from src.runner_set_objects import SetObjects
from src.runner_set_metrics import SetMetrics
from src.tools import Tools
from src.run import Run


class Runner(SetMetrics, SetObjects):
    def __init__(self, run: Run, rank, eval: bool = False):
        assert isinstance(run, Run), "run must be an instance of Run class"
        assert isinstance(rank, int), "rank must be an integer"
        self.run = run
        self.rank = rank
        self.t = Tools(DEBUG=DEBUG)
        self.eval = eval

        self.EARLY_BREAK = check_early_break(self.run.config.get('EARLY_BREAK', None))
      

        SetMetrics.__init__(self, run, rank)
        SetObjects.__init__(self, run, rank)
    
    @property
    def lew(self):
        return self.run.get_lew()


    def epoch_train(self, epoch):
        assert isinstance(epoch, int), "epoch must be an integer"
        self.eval = False  

        since = time.time()
        self.model.train()

        self.metrics['train'].reset()  # Reset metrics for the training epoch
  
        trn_loss = 0

        self.dataloader_train.sampler.set_epoch(epoch)

        for i, data in enumerate(self.dataloader_train): 

            if self.EARLY_BREAK is not None and i >= self.EARLY_BREAK['batches']:
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

            output = output.detach()  # Detach output to avoid tracking gradients
            self.metrics['train'].update(output, targets)  # Update training metrics

            del inputs, targets, output, loss

        trn_loss /= len(self.dataloader_train)

        # Sync and aggregate the loss across all processes
        aggr_loss_tensor = torch.tensor(trn_loss, device=f"cuda:{self.rank}")  # use the correct device
        dist.all_reduce(aggr_loss_tensor, op=dist.ReduceOp.SUM)
        trn_loss_mean = aggr_loss_tensor.item() / dist.get_world_size()
        logging.warning(f"rank {self.rank}: trn_loss: {trn_loss} trn_loss_mean: {trn_loss_mean}")

        self.scheduler.step(trn_loss_mean)

        self.metrics['train'].compute()

        elapsed = time.time() - since

        return elapsed, trn_loss_mean



    def epoch_eval(self, epoch, idx):
        assert isinstance(epoch, int), "epoch must be an integer"
        assert isinstance(idx, str), "idx must be a string"

        self.eval = True

        since = time.time()
        self.model.eval()

        self.metrics[idx].reset()  # Reset metrics for the validation epoch
       

        val_loss = 0

        
        self.dataloader_eval.sampler.set_epoch(epoch)

        with torch.no_grad():       
            for i, data in enumerate(self.dataloader_eval):

                if self.EARLY_BREAK is not None and i >= self.EARLY_BREAK['batches']:
                        logging.warning(f"rank {self.rank}: Eval Early break at batch {i}")
                        break
                    

                inputs = data[0].to(self.rank)
                targets = data[1].to(self.rank)

                self.t.memprint(f"rank {self.rank} batch {i} inputs {inputs.shape}, targets {targets.shape}")

                if self.patcher is not None:
                    logging.info(f"rank {self.rank} using patcher")
                    output = self.patcher(inputs)
                else:
                    output = self.model(inputs)

                self.t.memprint(f"rank {self.rank} outputs {output.shape}, targets {targets.shape}")

                loss = self.criterion(output, targets)
                val_loss += loss.item()            

                output = output.detach()  # Detach output to avoid tracking gradients
                self.metrics[idx].update(output, targets)



                del inputs, targets, output, loss

        val_loss /= len(self.dataloader_eval)

        # Sync and aggregate the loss across all processes
        aggr_loss_tensor = torch.tensor(val_loss, device=f"cuda:{self.rank}")  # use the correct device
        dist.all_reduce(aggr_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_mean = aggr_loss_tensor.item() / dist.get_world_size()
        logging.warning(f"rank {self.rank}: val_loss: {val_loss} val_loss_mean: {val_loss_mean}")


        self.metrics[idx].compute()# Update validation metrics

        elapsed = time.time() - since

        return elapsed, val_loss_mean


    def loop(self):

        self.set_metrics('train')
        self.set_metrics('val')

        for epoch in range(self.lew + 1, self.run.config["epochs"] + 1):

            if self.EARLY_BREAK is not None and epoch >= self.EARLY_BREAK['epochs']:
                    logging.info(f"rank {self.rank}: Early break at epoch {epoch}")
                    break

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

            logging.info(f'rank {self.rank}: Starting Validation Epoch: {epoch}')

            elapsedv, lossv = self.epoch_eval(epoch, 'val')

            if self.rank == 0:
                self.metrics['val'].save(epoch, elapsedv, lossv)


    def loop_eval(self, idx='test'):
        assert isinstance(idx, str), "idx must be a string"

        self.set_metrics(idx)
        

        epochs = tuple(self.run.config["eval_epochs"])

        assert len(epochs) == 2, "eval_epochs must be a tuple of two integers (start, end)"

        for epoch in range(epochs[0], epochs[1] + 1):

            if self.EARLY_BREAK is not None and epoch >= self.EARLY_BREAK['epochs']:
                logging.info(f"rank {self.rank}: Early break at epoch {epoch}")
                break

            self._set_model(epoch=epoch)

            logging.info(f'rank {self.rank}: Starting {idx} Epoch: {epoch}')

            elapsedv, lossv = self.epoch_eval(epoch, idx)

            if self.rank == 0:
                self.metrics[idx].save(epoch, elapsedv, lossv)



def check_early_break(EARLY_BREAK):
    if EARLY_BREAK is None:
        return None
    else:
        if "epochs" in EARLY_BREAK and "batches" in EARLY_BREAK:
            assert isinstance(EARLY_BREAK['epochs'], int), "EARLY_BREAK['epochs'] must be an integer"
            assert isinstance(EARLY_BREAK['batches'], int), "EARLY_BREAK['batches'] must be an integer"
            assert EARLY_BREAK['epochs'] > 0, "EARLY_BREAK['epochs'] must be greater than 0"
            assert EARLY_BREAK['batches'] > 0, "EARLY_BREAK['batches'] must be greater than 0"
        return EARLY_BREAK
