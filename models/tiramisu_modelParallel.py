import torch
import torch.nn as nn
import psutil
import os
process = psutil.Process(os.getpid())

from .layers import *

import time

def memprint(text):
    # Get total RAM used by the system
    total_ram_used = psutil.virtual_memory().used
    total_ram_used_gb = total_ram_used / (1024 ** 3)

    # Get the number of GPUs
    device_count = torch.cuda.device_count()
    gpu1_mem_used = torch.cuda.memory_allocated(0)/ (1024 ** 3)
    if (device_count>1):
        gpu2_mem_used = torch.cuda.memory_allocated(1)/ (1024 ** 3)
        print(text+f"\n {total_ram_used_gb:.2f} / {gpu1_mem_used:.2f} / {gpu2_mem_used:.2f}")
    else: 
        print(text+f"\n {total_ram_used_gb:.2f} / {gpu1_mem_used:.2f}")
    



class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12,
                 devices=['cuda:0', 'cuda:1', 'cpu'], split_indices=(1, 3)):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        num_down_blocks = len(down_blocks)
        cur_channels_count = 0
        skip_connection_channel_counts = []

        num_up_blocks = len(up_blocks)
        self.devices = devices
        self.split_indices = split_indices  # (x, y)
        x, y = split_indices

        ## First Convolution ##
        self.firstconv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_chans_first_conv, kernel_size=3,
                                   stride=1, padding=1, bias=True).to(devices[0])
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            device_idx = self.get_device_index(i, num_down_blocks)
            device = devices[device_idx]
            dense_block = DenseBlock(cur_channels_count, growth_rate, down_blocks[i]).to(device)
            self.denseBlocksDown.append(dense_block)
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            trans_down = TransitionDown(cur_channels_count).to(device)
            self.transDownBlocks.append(trans_down)

        #####################
        #     Bottleneck    #
        #####################

        device_idx = self.get_device_index('bottleneck', num_down_blocks)
        device = devices[device_idx]
        self.bottleneck = Bottleneck(cur_channels_count, growth_rate, bottleneck_layers).to(device)
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)):
            device_idx = self.get_device_index(num_up_blocks - i - 1, num_up_blocks, up=True)
            device = devices[device_idx]
            trans_up = TransitionUp(prev_block_channels, prev_block_channels).to(device)
            self.transUpBlocks.append(trans_up)
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            dense_block_up = DenseBlock(cur_channels_count, growth_rate, up_blocks[i],
                                        upsample=(i != len(up_blocks) - 1)).to(device)
            self.denseBlocksUp.append(dense_block_up)
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final Convolution ##
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True).to(devices[0])  # Assuming output on device[0]
        self.softmax = nn.LogSoftmax(dim=1)

    def get_device_index(self, layer_idx, total_layers, up=False):
        x, y = self.split_indices
        if (layer_idx == 'bottleneck'):
            return 2
        # if up:
        #     if layer_idx >= total_layers - x:
        #         return 0
        #     elif layer_idx >= total_layers - y:
        #         return 1
        #     else:
        #         return 2
        # else:
        if layer_idx <= x:
            return 0
        elif layer_idx <= y:
            return 1
        else:
            return 2

    def forward(self, x):
        skip_connections = []
        out = self.firstconv(x.to(self.devices[0]))
        memprint("first conv")


        # Downsampling path
        for i in range(len(self.down_blocks)):
            device_idx = self.get_device_index(i, len(self.down_blocks))
            device = self.devices[device_idx]
            out = out.to(device)
            memprint(f'+++ passed to                *** {device}')
            out = self.denseBlocksDown[i](out)
            memprint("down: "+str(i)+" dense block")
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            memprint("down: "+str(i)+" trans down")


        # Bottleneck
        device_idx = self.get_device_index('bottleneck', len(self.down_blocks))
        device = self.devices[device_idx]
        out = out.to(device)
        memprint(f'+++ passed to             *** {device}')
        out = self.bottleneck(out)
        memprint("bottleneck: "+str(i)+" bottlenck")


        # Upsampling path
        for i in range(len(self.up_blocks)):
            device_idx = self.get_device_index(len(self.up_blocks) - i - 1, len(self.up_blocks), up=True)
            device = self.devices[device_idx]
            skip = skip_connections.pop()
            out = out.to(device)
            memprint(f'+++ passed to            *** {device}')
           # print(f"skip device: {skip.device} passing to {device}")
            skip = skip.to(device)
            out = self.transUpBlocks[i](out, skip)
            memprint("up:"+str(i)+" trans conv")
            out = self.denseBlocksUp[i](out)
            memprint("up:"+str(i)+" dense block")


        # Final Convolution
        print(f"+++ passed to           *** {self.devices[0]}")
        out = out.to(self.devices[0])
        memprint(f'+++ passed to            *** {self.devices[0]}')
        out = self.finalConv(out)
        memprint("final conv")
        out = self.softmax(out)
        memprint("softmax")
        
        return out

# Instantiate the model with device assignments
def FCDenseNet67(n_classes, devices=['cuda:0', 'cuda:1', 'cpu'], split_indices=(1, 3)):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,
        devices=devices, split_indices=split_indices)


def FCDenseNet103(n_classes, devices=['cpu', 'cuda:0', 'cuda:1'], split_indices=(1, 3)):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,
        devices=devices, split_indices=split_indices)
