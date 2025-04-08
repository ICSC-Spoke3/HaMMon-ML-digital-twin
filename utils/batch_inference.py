# TO ADD ERROR HANDLING

import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(root_dir)
from pathlib import Path

import torch
from PIL import Image

from models import tiramisu_nclasses
from src import tools

from datasets.inference_dataset import InferenceDataset
from datasets.floodnet import Dataset

import time


# -------------------------------------------------------  OPTIONS
base_mem, one_img = 3.5, 3 # GB 
input_device = 'gpu' # 'gpu' 'auto'
weights_file = Path(root_dir) / '.weights' / 'fn_tl' / 'weights-200.pth'


input_folder='/outputs/airflow_data/floodnet/img-700'
output_folder = '/outputs/airflow_data/outputs'

output_folder = Path(output_folder)
output_folder.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------  FUNCTIONS

def gpu_index(mem):
    gpu_mem = mem[1:]
    return  max(range(len(gpu_mem)), key=lambda i: gpu_mem[i])+1


def set_device(input_device, mem):
    
    if (input_device == 'cpu') or len(mem) == 1: 
        return torch.device('cpu'), int(mem[0] // one_img)
    elif input_device in ('gpu', 'auto'):
        if len(mem) == 1:
            print("no gpu found, using cpu")
            return torch.device('cpu'), int(mem[0] // one_img)
        else:
            i = gpu_index(mem)
            # print(i, mem)
            return torch.device(f'cuda:{i-1}'), (int(mem[i])// one_img)

def save_segmentation_mask(tensor, path):
    array = tensor.cpu().numpy().astype('uint8')
    img = Image.fromarray(array, mode='L')  # 'L' = 8-bit grayscale
    img.save(path)


# -------------------------------------------------------  MAIN

def main():
    

    # choose device and batch size
    mems = tools.free_mem()
    device, batch_size = set_device(input_device, mems)

    infererence_dataset = InferenceDataset(Dataset_model=Dataset, dataset_folder=input_folder)

    print(f"Processing device: {device}, Batch size: {batch_size}, Dataset size: {len(infererence_dataset)}")

    weights = torch.load(weights_file, map_location=device)
    model = tiramisu_nclasses.FCDenseNet103(len(infererence_dataset.class_names)).to(device)
    model.load_state_dict(weights['state_dict'])

    dataloader = torch.utils.data.DataLoader(infererence_dataset, batch_size=batch_size, shuffle=False)

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():

        start = time.time()
        for batch_i, (batch, indexes) in enumerate(dataloader):

            # if batch_i > 0:
            #     break

            batch = batch.to(device)
            output = model(batch)

            pred = tools.get_predictions(output)

            for i, t in enumerate(pred):
                path = infererence_dataset.imgs[indexes[i]]
                out_path = output_folder / path.with_suffix('.png').name                
                save_segmentation_mask(t, out_path)
        
        print(f"Dataset processed in {time.time() - start:.2f} seconds")
            
main()