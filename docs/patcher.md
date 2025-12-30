# Patcher

## Overview
`patcher.py` provides a small utility to run inference on large tensors by splitting them into patches, applying a model to each patch, and stitching the results back together. It is designed for tensor-level operations (batches of model inputs), not for high-level image objects.

This is useful when a full input does not fit in GPU memory or when you want controlled patch-wise inference with configurable kernel and stride.

```python 
import torch
from src.patcher import Patcher

# dummy model
model = MyModel()

# create patcher
patcher = Patcher(
    model=model,
    kernel=(256, 256),
    stride=(128, 128),
    device="cuda:0",
    mode="average"
)

# input tensor (B, C, H, W)
x = torch.randn(1, 3, 1024, 1024)

# run patch-based inference
with torch.no_grad():
    y = patcher(x)  # y is the recomposed output tensor
```

## What it does
- Takes a 4D input tensor (B, C, H, W) and computes a patching layout.
- Extracts overlapping patches with a chosen kernel and stride.
- Runs the model on each patch independently.
- Recombines patch outputs into a full-sized tensor.
- Aggregates overlapping regions by averaging logits (the default mode).

## How it fits into the loop
The patcher can be used for validation/test inference. When `self.patcher` is set, the evaluation loop in `Runner.epoch_eval()` uses the patcher instead of calling the model directly. This keeps training unchanged while allowing patch-based inference during validation and test.

Example setup in a runner:

```python
# inside your runner init
def set_patcher(self):
    self.patcher = Patcher(
        model=self.model,
        kernel=(64,64),
        stride=(32,32),
        device=self.rank
    )
```

## Typical use case
- Large spatial inputs that do not fit end-to-end on the GPU.
- Need for sliding-window inference with overlap control.
- Evaluation or test runs where patching is desirable without changing training.

## Disclaimer
This tool has been used in specific projects and is not designed or tested for general-purpose use. In some configurations or edge cases, unexpected behavior may occur; use it with caution and validate outputs in your context.
