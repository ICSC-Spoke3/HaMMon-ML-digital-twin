"""
Runs batch segmentation inference over an image folder and saves the predicted masks.

Disclaimer: this script is intended only to validate the Science Gateway pipeline;
it is not a production-grade inference entrypoint. Error handling is not managed.
Model selection, weights path, dataset choice, and other runtime settings should be
exposed as command-line options rather than hardcoded.

example usage: 
> python batch_inference.py /input_folder /output_folder

"""


import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, str(root_dir)) 
from pathlib import Path
import importlib
import argparse
import time

import torch
from PIL import Image


from src.tools import Tools
from src.predict import Predict

from datasets.inference_dataset import InferenceDataset

tools = Tools(DEBUG=True)
predict = Predict(kind='max')

# -------------------------------------------------------  TO HANDLE AS OPTIONS

input_device = 'auto' # 'cpu', 'gpu' 'auto'
dataset_model = 'rescuenet'

## model 
from models.att_unet import sAttU_Net
from models.stiramisu import sFCDenseNet103
weights_file = Path(root_dir) / '.weights' / 'stiramisu-rn.pth'
one_img = 3 # GB , memory required to compute the model with one image

## Dataset (to get classes, mean, std)
module = importlib.import_module(f"datasets.{dataset_model}")
Dataset = getattr(module, 'Dataset')
Dataset.stats_from_yaml(f'{dataset_model}.yaml')


# -------------------------------------------------------  FUNCTIONS

def gpu_index(mem):
    gpu_mem = mem[1:]
    return  max(range(len(gpu_mem)), key=lambda i: gpu_mem[i])+1

def compute_bs(value):
    # Calculate batch size and ensure it's at least 1 and value is non-negative
    if value < 0:
        raise ValueError("Negative memory value is invalid")
    batch_size = int(value // one_img)
    if batch_size < 1:
        raise ValueError("Insufficient memory to process at least one image")
    return batch_size

def set_device(input_device, mem):
    
    if (input_device == 'cpu'): 
        return torch.device('cpu'), compute_bs(mem[0])  # Will raise error if invalid
    elif input_device in ('gpu', 'auto'):
        if len(mem) == 1:
            print("no gpu found, using cpu")
            return torch.device('cpu'), compute_bs(mem[0])
        else:
            try:
                # Select the best GPU based on memory
                i = gpu_index(mem)
                batch_size = compute_bs(mem[i])
                return torch.device(f'cuda:{i-1}'), batch_size
            except ValueError as e:
                # If GPU memory is insufficient or invalid, fall back to CPU
                print(f"GPU memory error ({mem[i]}): {e}. Falling back to CPU...")
                return set_device('cpu', mem)

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

    weights = torch.load(weights_file, map_location=device)['data']
    model = sFCDenseNet103(len(infererence_dataset.class_names)).to(device)
    model.load_state_dict(weights)

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

            pred = predict(output)

            for i, t in enumerate(pred):
                path = infererence_dataset.imgs[indexes[i]]
                out_path = output_folder / path.with_suffix('.png').name                
                save_segmentation_mask(t, out_path)
        
        print(f"Dataset processed in {time.time() - start:.2f} seconds")
            

# -------------------------------------------------------  PARSE ARGUMENTS
parser = argparse.ArgumentParser(description='Segmentation inference script')
parser.add_argument('input_folder', type=str, help='Path to the input image folder')
parser.add_argument('output_folder', type=str, help='Path to the output folder for masks')
args = parser.parse_args()

input_folder = Path(args.input_folder)
output_folder = Path(args.output_folder)

if not input_folder.exists() or not any(input_folder.iterdir()):
    raise FileNotFoundError(f"Input folder '{input_folder}' does not exist or is empty.")

output_folder.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    main()


