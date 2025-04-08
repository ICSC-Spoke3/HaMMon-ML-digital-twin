import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
import psutil

def mem():

    total_ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    device_count = torch.cuda.device_count()

    if device_count == 0:
        # Return only RAM usage
        return (total_ram_used_gb,)
    else:
        # Get GPU memory usage for each GPU
        gpu_mems_used_gb = [torch.cuda.memory_allocated(i) / (1024 ** 3) for i in range(device_count)]
        # Return RAM usage and GPU memory usages
        return (total_ram_used_gb,) + tuple(gpu_mems_used_gb)


def total_mem():
    """Returns (total CPU RAM, total GPU1 RAM, total GPU2 RAM, ...) in GiB"""
    total_ram_cpu_gb = psutil.virtual_memory().total / (1024 ** 3)
    device_count = torch.cuda.device_count()

    if device_count == 0:
        return (total_ram_cpu_gb,)
    total_gpu_mems_gb = [
        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        for i in range(device_count)
    ]
    return (total_ram_cpu_gb,) + tuple(total_gpu_mems_gb)

def free_mem():
    """Returns (free CPU RAM, free GPU1 RAM, free GPU2 RAM, ...) in GiB"""
    free_ram_cpu_gb = psutil.virtual_memory().available / (1024 ** 3)
    device_count = torch.cuda.device_count()

    if device_count == 0:
        return (free_ram_cpu_gb,)
    free_gpu_mems_gb = [
        (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)
        for i in range(device_count)
    ]
    return (free_ram_cpu_gb,) + tuple(free_gpu_mems_gb)

def memprint(text):
    mem_results = mem()
    mem_strings = " / ".join(f"{mem:.2f}" for mem in mem_results)
    print(text + f"\n {mem_strings}")

def get_predictions(output_batch):
    # "c" is the number of channels=classes
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.max(1)
    indices = indices.view(bs,h,w)
    return indices