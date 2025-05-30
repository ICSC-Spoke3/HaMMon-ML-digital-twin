import time
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.autograd import Variable
 
from torchmetrics import JaccardIndex as IoU


DEBUG = False
if not DEBUG: 
    memprint = lambda x : x 
else:
    from .tools import memprint


# ----------------------------------------------------------------------  utilities

def get_predictions(output_batch):
    # "c" is the number of channels=classes
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return err

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()

def inverse_frequency_weights(n):
    arr = torch.tensor(n, dtype=torch.float32)
    frequencies = arr / arr.sum()
    weights = 1 / frequencies  # Inverse of the frequencies
    return weights

def median_frequency_weights(n):
    arr = torch.tensor(n, dtype=torch.float32)
    frequencies = arr / arr.sum()
    median_freq = torch.median(frequencies)  # Get the median of the frequencies
    weights = median_freq / frequencies      # Median frequency divided by each frequency
    return weights

# ----------------------------------------------------------------------  training

def train(model, trn_loader,num_classes, optimizer, criterion, device):
    model.train()
    trn_loss = 0
    trn_error = 0
    iou = IoU(task='multiclass', num_classes=num_classes, average="none").to(device)

    for idx, data in enumerate(trn_loader):
    # progress_bar = tqdm(enumerate(trn_loader), total=len(trn_loader), desc=f"Training Epoch {epoch}")
    # for idx, data in progress_bar:


        # if idx == 10:
        #     break 

        memprint('--------------------------------------- batch '+str(idx)+"    ")
        #print(f"batch {idx}")
        torch.cuda.empty_cache()

        inputs = data[0].to(device)
        targets = data[1].to(device)

        memprint(f" inputs {inputs.shape}, targets {targets.shape}")

        memprint('pre-calculation')       
        output = model(inputs)
        memprint('after output')

        memprint(f" outputs {output.shape}, targets {targets.shape}")
        loss = criterion(output, targets)
        trn_loss += loss.data.item()
        loss.backward()

        memprint('after backprop')

        optimizer.step()
        optimizer.zero_grad()           
        memprint('after step') 

   
        pred = get_predictions(output)
        iou.update(pred, targets)                               ### IOU 

        trn_error += error(pred, targets.data)

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    iou_result = iou.compute()
    iou.reset()
    return trn_loss, trn_error, iou_result


# ----------------------------------------------------------------------  test

def test(model, test_loader, num_classes, criterion, device):
    since = time.time()

    model.eval()
    test_loss = 0
    test_error = 0
    i=0
    print('test')


    memprint("before iterat")
    iou = IoU(task='multiclass', num_classes=num_classes, average="none").to(device)

    #for data, target in tqdm(test_loader, desc="Processing"):

    for data, target in test_loader:
        with torch.no_grad():
            
            #print(i, data.shape, target.shape)
            torch.cuda.empty_cache()
            
            i +=1

            # if (i>1):
            #     break
            
            target = target.to(device)        
            data = data.to(device)
            
            memprint("before model eval, batch: {}".format(i))
            output = model(data)
            memprint("after output") 
            # print('+', output.device, target.device)
            crit=criterion(output, target)
            #print("crit.shape", crit.shape)
            test_loss += crit.item()
            pred = get_predictions(output)

            test_error += error(pred, target).item()
    
            iou.update(pred, target)                               ### IOU 

            memprint("end of cycle")
            time_elapsed = time.time() - since  
            memprint('Batch {} Total Time {:.0f}m {:.0f}s\n'.format(i, time_elapsed // 60, time_elapsed % 60))
            # print('Batch {} Total Time {:.0f}m {:.0f}s\n'.format(i, time_elapsed // 60, time_elapsed % 60))
        
            # print(f"Allocated memory after batch {i}: {torch.cuda.memory_allocated() / 1e6} MB")
            # print(f"Cached memory after batch {i}: {torch.cuda.memory_reserved() / 1e6} MB")
            # print(torch.cuda.memory_summary(device=0, abbreviated=True))

            torch.cuda.empty_cache()
            #print(f"Cached memory after batch {i}: {torch.cuda.memory_reserved() / 1e6} MB")



    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    iou_result = iou.compute()
    iou.reset()

    return test_loss, test_error, iou_result


