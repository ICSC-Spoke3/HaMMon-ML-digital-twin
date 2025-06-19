import torch
import random
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import logging


from src.settings import load_settings
from src.run import Run


#################################### Logging and debug modules

import logging
# different level of logging can be set here, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)



#################################### DDP setup

def ddp_setup(rank, settings):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = settings["MASTER_ADDR"]
    os.environ["MASTER_PORT"] = settings["MASTER_PORT"]

    torch.cuda.set_device(rank)
    
    init_process_group(backend=settings["backend"], rank=rank, world_size=settings["world_size"])


#################################### Random seed setting

def set_seed(seed: int):
    '''
    Set random seed for reproducibility.
    '''

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#################################### Main function for distributed training

def main(rank: int, run: Run, Runner, settings: dict, testname: str):

    logging.info(f'rank {rank} started')

    ddp_setup(rank, settings)
    set_seed(42 + rank)  

    runner = Runner(rank=rank, run=run)

    if testname is None:
        runner.loop()
    else:
        runner.loop_eval(idx=testname)

    destroy_process_group()


#################################### Main entry point for the script

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='distributed training job')
    parser.add_argument('run', type=str, help='run name')
    parser.add_argument('--clear', action='store_true', help='clear previous data')
    parser.add_argument('--test', type=str, help='name of the test')
    args = parser.parse_args()

    run = Run(args.run)

    if args.clear:
        #-------------- comment to disable the check
        confirm = input("Are you sure you want to clear data? [y/n]: ").strip().lower()
        if confirm != 'y':
            logging.info("Clear operation cancelled.")
            exit(0)
        #------------------------------------------
        run.clear(weights=True)
        logging.info(f"run {args.run} cleared")
        exit(0)

    settings = load_settings()
    logging.info(f"Settings loaded: {settings}")

    sys.path.append(str(run.folder))

    if args.test is None:
        logging.info("Running in training mode")
        from runner_init import TrainRunner as Runner
    else:
        logging.info("Running in test mode")
        from runner_init import TestRunner as Runner

    mp.spawn(main, args=(run, Runner, settings, args.test), nprocs=settings["world_size"])
