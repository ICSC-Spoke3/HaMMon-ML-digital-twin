import torch


class Predict:
    def __init__(self, kind: str):
        assert isinstance(kind, str), f"prediction_method must be a string, but got {type(kind)}"
        self.implemented = ['max', 'sigmoid', 'threshold', 'binary_predictions']
        assert kind in self.implemented, f"prediction_method must be one of {self.implemented}, but got {kind}"
        self.kind = kind

    def __call__(self, output, **kwargs):
        """
        Apply the specified prediction method to the output tensor.
        
        Args:
            output (torch.Tensor): The model's output tensor.
            
        Returns:
            torch.Tensor: The processed prediction tensor.
        """
        assert isinstance(output, torch.Tensor), f"output must be a torch.Tensor, but got {type(output)}"

        match self.kind:
            case 'max':
                return self.max_label(output)
            case 'sigmoid':
                return self.binary_sigmoid(output)
            case 'threshold':
                return self.threshold(output)
            case 'binary_predictions':
                return self.binary_predictions(output, **kwargs)
            case _:
                raise RuntimeError(f"Prediction kind '{self.kind}' is not implemented. We should not be here.")



    def max_label(self, output, axis=1):
        """
        Get the index of the maximum value along a specified axis.

        Args:
            preds (torch.Tensor): Input tensor.
            axis (int): Axis along which to find the maximum. Default is 1.
            
        Returns:
            torch.Tensor: Indices of the maximum values along the specified axis.
        """
        return output.argmax(dim=axis)
    
    def binary_sigmoid(self, output):
        """
        Apply sigmoid activation to the difference of channels 1 and 0 of 
        the output tensor.
        It is the probability of the class 1.

        Args:
            output (torch.Tensor): The model's output tensor.
            
        Returns:
            torch.Tensor: Sigmoid activated tensor.

        """
        (B, C, H, W) = output.shape
        if C != 2:
            raise ValueError(f"Expected output with 2 channels for binary classification, but got {C} channels.")
        
        diff = output[:, 1, :, :] - output[:, 0, :, :]
        return torch.sigmoid(diff)
    
    def threshold(self, output, threshold):
        """
        Apply a threshold to the output tensor for binary classification.

        Args:
            output (torch.Tensor): The model's output tensor.
            threshold (float): Threshold value to apply.
            
        Returns:
            torch.Tensor: Binary tensor after applying the threshold.
        """
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be between 0 and 1, but got {threshold}.")
        
        return (output > threshold).float()
    
    def binary_predictions(self, output, threshold):
        """
        Get binary predictions based on the output tensor and a threshold.

        Args:
            output (torch.Tensor): The model's output tensor.
            threshold (float): Threshold value to apply.

        Returns:
            torch.Tensor: Binary predictions after applying the threshold.
        """
        output = self.binary_sigmoid(output)    # get the probabiity for class 1
        return self.threshold(output, threshold)
    





