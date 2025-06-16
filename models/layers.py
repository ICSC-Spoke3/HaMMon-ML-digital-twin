import torch
import torch.nn as nn
import psutil
import os
process = psutil.Process(os.getpid())


def memprint(text):
    #   gpu part
    #print(text.ljust(30)+"%fMB"%(torch.cuda.memory_allocated(0)/1024/1024) +"   %fMB"%(torch.cuda.memory_allocated(1)/1024/1024))
    
    # #   psutil part
    # if (text): print(text)
    # mem = psutil.virtual_memory()
    # # Fetch system-wide cached memory
    # cached_memory = mem.cached
    # # Fetch system-wide buffered memory
    # buffered_memory = mem.buffers
    # # Calculate adjusted used memory
    # adjusted_used_memory = mem.used + cached_memory + buffered_memory
    # used = mem.used / (1024 ** 3)
    # # used by process/system/+cached
    # print(f"RAM {process.memory_info().rss / 1024 ** 3:.2f} GB {used:.2f} GB {adjusted_used_memory / (1024 ** 3):.2f} GB")
    
    return

def iddi():
    return str(torch.randint(100, 1000, (1,)).item())

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    # def forward(self, x, iddi):
    #     return super().forward(x)
    def forward(self, x, iddi):
        for name, module in self.named_children():
            x = module(x)
            memprint(name+" "+ iddi)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        id = iddi()
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            memprint("DenseBlock(up) "+id)
            for layer in self.layers:
                out = layer(x, id)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            memprint("DenseBlock(down)"+id)
            for layer in self.layers:
                out = layer(x, id)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]
