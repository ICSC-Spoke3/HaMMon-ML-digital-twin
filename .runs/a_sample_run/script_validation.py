
import os
import sys
import uuid
import time
from pathlib import Path

root_dir = Path(os.getcwd()).resolve().parent.parent
sys.path.append(str(root_dir))

import time

import torch
import torch.nn as nn


from models import tiramisu
from datasets import joint_transforms
import utils.training_noAcc as train_utils
from utils.experiment import Experiment

print(time.ctime(time.time()))
exp = Experiment('corr_new_noAccum_adam_sched')
# exp.clear()
# sys.exit()


crop_size_eval = tuple(map(int, crop_size_eval.split(', ')))  # Split by comma and space, then map to integers


dataset = exp.Dataset(
    split='val', 
    joint_transform=joint_transforms.FixedUpperLeftCrop(crop_size_eval)
    )
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=exp.config['batch_size_eval'], shuffle=False)

torch.cuda.manual_seed(0)
device = torch.device("cuda") 
model = tiramisu.FCDenseNet103(len(dataset.class_names))
model.cuda()
fw = train_utils.inverse_frequency_weights(dataset.pixel_count)
class_weights =  torch.log(fw).cuda()
criterion = nn.NLLLoss(weight=class_weights)


last_epoch= exp.get_last_epoch_in_results('val')
START_EPOCH = 0 if last_epoch==None else last_epoch+1


selected_epoch = START_EPOCH



try:
    exp.load_weights(model, selected_epoch)
    model = nn.DataParallel(model, device_ids=[0,1])
except Exception as e:
    print(f"Error loading weights: {e}")
    sys.exit()

print(f'START_EPOCH {START_EPOCH:d}')

since = time.time()
num_classes = len(dataset.class_names)

### Test ###

loss, err, iou = train_utils.testDP(model, dataloader, num_classes, criterion, device)    
print('Val - Loss: {:.4f} | Err: {:.4f}'.format(loss, err))
time_elapsed = time.time() - since  
print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))


exp.save_results('val', selected_epoch, time_elapsed, loss, err, iou)



