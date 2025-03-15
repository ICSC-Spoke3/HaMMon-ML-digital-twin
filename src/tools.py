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

def memprint(text):
    mem_results = mem()
    mem_strings = " / ".join(f"{mem:.2f}" for mem in mem_results)
    print(text + f"\n {mem_strings}")

