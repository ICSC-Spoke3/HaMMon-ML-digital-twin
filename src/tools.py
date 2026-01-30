import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


class Tools:
    def __init__(self, DEBUG):
        assert isinstance(DEBUG, bool), "DEBUG must be a boolean value"
        self.DEBUG = DEBUG
        if self.DEBUG:
            import torch
            self.torch = torch
            import psutil
            self.psutil = psutil


    def mem(self):

        assert self.DEBUG, "mem() can only be called when DEBUG is True"

        total_ram_used_gb = self.psutil.virtual_memory().used / (1024 ** 3)
        device_count = self.torch.cuda.device_count()

        if device_count == 0:
            # Return only RAM usage
            return (total_ram_used_gb,)
        else:
            # Get GPU memory usage for each GPU
            gpu_mems_used_gb = [self.torch.cuda.memory_allocated(i) / (1024 ** 3) for i in range(device_count)]
            # Return RAM usage and GPU memory usages
            return (total_ram_used_gb,) + tuple(gpu_mems_used_gb)

    def memprint(self, text):

        if self.DEBUG:
            mem_results = self.mem()
            mem_strings = " / ".join(f"{mem:.2f}" for mem in mem_results)
            logging.info(text + f"\n {mem_strings}")
        else:
            return 


# def mem():

#     total_ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
#     device_count = torch.cuda.device_count()

#     if device_count == 0:
#         # Return only RAM usage
#         return (total_ram_used_gb,)
#     else:
#         # Get GPU memory usage for each GPU
#         gpu_mems_used_gb = [torch.cuda.memory_allocated(i) / (1024 ** 3) for i in range(device_count)]
#         # Return RAM usage and GPU memory usages
#         return (total_ram_used_gb,) + tuple(gpu_mems_used_gb)


    def total_mem(self):
        """Returns (total CPU RAM, total GPU1 RAM, total GPU2 RAM, ...) in GiB"""
        total_ram_cpu_gb = self.psutil.virtual_memory().total / (1024 ** 3)
        device_count = self.torch.cuda.device_count()

        if device_count == 0:
            return (total_ram_cpu_gb,)
        total_gpu_mems_gb = [
            self.torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(device_count)
        ]
        return (total_ram_cpu_gb,) + tuple(total_gpu_mems_gb)

    def free_mem(self):
        """Returns (free CPU RAM, free GPU1 RAM, free GPU2 RAM, ...) in GiB"""
        free_ram_cpu_gb = self.psutil.virtual_memory().available / (1024 ** 3)
        device_count = self.torch.cuda.device_count()

        if device_count == 0:
            return (free_ram_cpu_gb,)
        free_gpu_mems_gb = [
            (self.torch.cuda.get_device_properties(i).total_memory - self.torch.cuda.memory_allocated(i)) / (1024 ** 3)
            for i in range(device_count)
        ]
        return (free_ram_cpu_gb,) + tuple(free_gpu_mems_gb)

