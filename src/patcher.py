import torch
import torch.nn.functional as F
import logging

class Patcher:
    """
    A class to apply a model to patches of an input tensor, with options for kernel size, stride, and padding.
    The output is then recombined into a single tensor, either by averaging or max pooling the results.
    Args:
        model (torch.nn.Module): The model to apply to each patch.
        kernel (int or tuple): The size of the kernel to use for patch extraction.
        stride (int or tuple): The stride to use when extracting patches.
        padding_mode (str, optional): The padding mode to use ('constant' or 'reflect'). Default is 'reflect'.
        device (str or torch.device, optional): The device to run the model on. Default is 'cuda' if available, otherwise 'cpu'.
        mode (str, optional): The mode of operation ('average' or 'max'). Default is 'average'.
    """

    def __init__(self, model, kernel, stride, padding_mode='reflect', device=None, mode='average', predict=None, debug=False):
        self.predict = predict

        assert isinstance(debug, bool), "Debug must be a boolean value."
        self.debug = debug
        
        if mode in ['average', 'max']:
            self.mode = mode
        else:
            raise ValueError("Mode must be either 'average' or 'max'.")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Invalid device argument: {device}")
  
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model must be an instance of torch.nn.Module.")
        self.model = model
        self.model = model.to(self.device)

        if isinstance(kernel, int):
            self.kernel = (kernel, kernel)
        elif isinstance(kernel, tuple) and len(kernel) == 2:
            self.kernel = kernel
        else:
            raise ValueError("Kernel must be an int or a tuple of two ints.")
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise ValueError("Stride must be an int or a tuple of two ints.")
        
        if any(k<s for k,s in zip(self.kernel, self.stride)):
            raise ValueError("Kernel size must be greater than or equal to stride size.") 

        if padding_mode  in ['constant', 'reflect']:
            self.padding_mode = padding_mode
        else:
            raise ValueError("Padding mode must be one of 'constant', 'reflect'")
        
    def compute_batch_layout(self, input_shape):
        """
        Computes the layout of the input tensor for batching.
        Args:
            input_shape (tuple): The shape of the input tensor.
        Returns:
            tuple: The batch layout.
        """
        if not isinstance(input_shape, tuple) or len(input_shape) != 4:
            raise ValueError("Input shape must be a tuple of length 4 (B, C, H, W).")
        
        B, C, H, W = input_shape
        k = self.kernel
        s = self.stride

        d = H -k[0], W - k[1]


        n = max(0, d[0] // s[0] + int(d[0] % s[0] != 0)), \
            max(0, d[1] // s[1] + int(d[1] % s[1] != 0))
        canvas = k[0] + s[0] * n[0], k[1] + s[1] * n[1]
        pad = canvas[0] - H, canvas[1] - W
        pad_left, pad_right = pad[1] // 2 + pad[1] % 2, pad[1] // 2
        pad_top, pad_bottom = pad[0] // 2 + pad[0] % 2, pad[0] // 2

        return {
            'n_steps': n,
            'patches': (n[0]+1, n[1]+1),  
            'input_size': (H, W),
            'canvas_size': canvas,
            'padding': pad,
            'pad_left': pad_left,
            'pad_right': pad_right,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom
        }

        

    def __call__(self, input):

        if input.dim() == 3:
            input = input.unsqueeze(0).to(self.device)  # Add batch dimension if missing
        B, C, H, W = input.shape
        input = input.to(self.device)

        d = self.compute_batch_layout(input.shape)


        k = self.kernel
        s = self.stride

        n = d['n_steps']  # number of steps
        canvas = d['canvas_size']  # canvas size after padding
        pad = d['padding']  # padding to make the input size divisible by the stride
        pad_left, pad_right = d['pad_left'], d['pad_right']
        pad_top, pad_bottom = d['pad_top'], d['pad_bottom']


        if self.padding_mode == 'constant':
            input = torch.nn.functional.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0).to(self.device)
        elif self.padding_mode == 'reflect':
            input = torch.nn.functional.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').to(self.device)


        # If labels, convert to float32
        if not torch.is_floating_point(input): 
            input = input.to(torch.float32)

        logging.debug(f'kernel: {k}, stride: {s}, steps: {n}, canvas size: {canvas}, padded input shape: {input.shape}, padding: {pad}, pad_left: {pad_left}, pad_right: {pad_right}, pad_top: {pad_top}, pad_bottom: {pad_bottom}')

        # Extract patches using unfold

        patches = F.unfold(input, kernel_size=k, stride=s) # (B, C*k*k, N)
        N = patches.shape[-1] # number of patches
        
                # -> (B, N, C*k*k)        # -> (B,N, C, k, k)        # -> (N, B, C, k[0], k[1])  
        patches = patches.transpose(1, 2).reshape(B, N, C, k[0], k[1]).transpose(0, 1)       


        # apply the model with the same bach size as the input
        output = []
        for i in range(0, N):
            o = self.model(patches[i]).unsqueeze(0)  # (B, C, k[0], k[1])
            output.append(o)
        output = torch.cat(output, dim=0) # (N, B, C, k[0], k[1])
        n_classes = output.shape[2]  # Number of classes

        logging.debug(f'unfolded shape: {patches.shape}, output shape: {output.shape}')


        # if (self.debug):
        #     # === DEBUG VISUALIZATION (optional, comment/uncomment and fix as needed) ===
        #     import sys
        #     from pathlib import Path
        #     from datasets.rescuenet_resized import Dataset
        #     from matplotlib import pyplot as plt
        #     import numpy as np
        #     # -------------------------------------------------------------------
        #     root_folder = Path(__file__).resolve().parent.parent
        #     sys.path.append(str(root_folder))
        #     from src.imgs import Img
        #     # -------------------------------------------------------------------
        #     img = Img(Dataset)
        #     # -------------------------------------------------------------------
        #     index = 0
        #     cols, rows = n[1]+1, n[0]+1
        #     fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), constrained_layout=True)
        #     axes = np.array(axes).reshape(rows, cols)

        #     for idx in range(N):
        #         ax = axes[idx // cols, idx % cols]
        #         # multiple class
        #         #img_np = img.label_to_np(output[idx][index].argmax(0).squeeze(0).cpu()) # (H, W)
        #         #single class
        #         img_np = img.label_to_np(self.predict(output[idx][index]).squeeze(0).cpu()) # (H, W)
        #         ax.imshow(img_np, cmap='gray')
        #         ax.axis('off')
        #     #plt.tight_layout()
        #     plt.show()
        #     # ===================================================================

        if self.debug:
            import matplotlib.pyplot as plt
            import numpy as np

            index = 0  # first image in batch
            cols, rows = n[1] + 1, n[0] + 1
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), constrained_layout=True)
            axes = np.array(axes).reshape(rows, cols)

            for idx in range(N):
                ax = axes[idx // cols, idx % cols]
                # Predict class mask for patch
                mask = self.predict(output[idx][index]).squeeze(0).cpu().numpy().astype(np.uint8)  # shape: (H, W)

                # Build custom colormap: class 0 -> black, class 1 -> white, others → assign colors automatically
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(['black', 'white'] + list(plt.cm.tab10.colors[2:]))

                ax.imshow(mask, cmap=cmap, vmin=0, vmax=10)
                ax.axis('off')

            plt.show()


        # combine results back to the original canvas size
        
        if self.mode == 'average':
            # using folding, average pooling of logits
                                # -> (B, N, C, k, k)    -> (B, N, C*k*k)    -> (B, C*k*k, N)
            output_patches_flat = output.transpose(0,1).reshape(B, N, -1).transpose(1, 2) 
            output = F.fold(output_patches_flat, output_size=canvas, kernel_size=k, stride=s)
            #==== NORMALIZZAZIONE ====
            ones = torch.ones_like(output_patches_flat)
            norm_map = F.fold(ones, output_size=canvas, kernel_size=k, stride=s)
            output = output / norm_map

        elif self.mode == 'max':
            raise NotImplementedError("Max pooling mode is not implemented yet.")
            # output = output.transpose(0, 1)  # (B, N, C, k[0], k[1])
            # output_full = torch.full((B, n_classes, canvas[0], canvas[1]), float('-inf'), device=input.device)

            # count = 0
            # for i in range(0, canvas[0] - k[0] + 1, s[0]):
            #     for j in range(0, canvas[1] - k[1] + 1, s[1]):
            #         patch = output[count]
            #         #patch = patch.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, kH, kW)
            #         output_full[:, :, i:i+k[0], j:j+k[1]] = torch.maximum(
            #             output_full[:, :, i:i+k[0], j:j+k[1]],
            #             patch
            #         )
            #         count += 1
            # output = output_full


        output =  output[:, :, pad_top:pad_top+H, pad_left:pad_left+W]
        # Convert to class indices
        return output  

