
import sys
import os
from pathlib import Path

root_dir = Path(os.getcwd()).resolve().parent.parent
sys.path.append(str(root_dir))

import time

import torch
import torch.nn as nn

import torchvision.transforms as transforms

from models import tiramisu
from datasets import joint_transforms
import utils.training_noAcc as train_utils
from utils.experiment import Experiment


exp = Experiment('corr_new_noAccum_adam_sched')


train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomCrop(exp.config['crop_size'), 
    joint_transforms.JointRandomHorizontalFlip()
    # joint_transforms.JointRandomRotate90(),
    # joint_transforms.JointRandomFlip()
    ])

dataset = exp.Dataset(split='train',scale=exp.config['dataset_scale'],
      joint_transform=train_joint_transformer) 

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=exp.config['batch_size'], shuffle=True)


torch.cuda.manual_seed(0)
device = torch.device("cuda") 
model = tiramisu.FCDenseNet103(len(dataset.class_names))
model.cuda()



lw = exp.get_last_weights()
if not lw: 
    print('initializing weights')
    model.apply(train_utils.weights_init)
    START_EPOCH = 0
else:
    exp.load_latest_weights(model)
    START_EPOCH = lw[0]+1
print(f'START EPOCH {START_EPOCH}')


model = nn.DataParallel(model, device_ids=[0,1])

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=exp.config['LR'], 
                             betas=(exp.config['beta1'], exp.config['beta2']),
                             eps=1e-08, 
                             weight_decay=exp.config['weight_decay'])
if START_EPOCH > 0:
    opt = exp.load_latest('optimizer')
    print(f"loaded optimizer of epoch {opt['startEpoch']}")
    optimizer.load_state_dict(opt['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = exp.config['weight_decay']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
if START_EPOCH > 0:
    sch = exp.load_latest('scheduler')
    print(f"loaded scheduler of epoch {sch['startEpoch']}")
    scheduler.load_state_dict(sch['scheduler'])



fw = train_utils.inverse_frequency_weights(dataset.pixel_count)
class_weights =  torch.log(fw).cuda()

criterion = nn.NLLLoss(weight=class_weights)
 

num_classes = len(dataset.class_names)

epoch = START_EPOCH

since = time.time()

if epoch <= 500:
    print('Starting Training Epoch:', epoch)
    # Train ###
    trn_loss, trn_err, iou_result = train_utils.train(
        model, dataloader, num_classes, optimizer, criterion, epoch, device, exp.config['ACCUMUL_SIZE'])

   # trn_loss, trn_err, iou_result = 1, 0.3, [1,2,3,4,5,6,7,8,9]

    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # saving LR to a csv for convenience
    lr = optimizer.param_groups[0]['lr']
    sched = scheduler.state_dict()
    #exp.save_epoch_data(epoch, 'LR', {'lr':lr, 'sched_lr': sched['_last_lr'], 'num_bad_epochs':sched['num_bad_epochs']  })
    exp.save_epoch_data(epoch, 'LR', {'lr':lr, 'sched': sched})

    # updating LR
    scheduler.step(trn_loss)

    ### Saving weights ###
    exp.save_weigths(model.module.state_dict(), epoch)
    exp.save_results('train', epoch, time_elapsed, trn_loss, trn_err, iou_result)
    exp.save('optimizer', epoch, optimizer.state_dict())
    exp.save('scheduler', epoch, scheduler.state_dict())


    
        



