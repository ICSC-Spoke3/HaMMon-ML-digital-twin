import torch


def inverse_frequency_weights(n, beta=1):
    """
    Calculate inverse frequency weights for a list of class counts.
    Args:
        n (list or array-like): List of class counts.
        beta (float): Exponent to adjust the weights. Default is 1.
    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    arr = torch.tensor(n, dtype=torch.float32)
    frequencies = arr / arr.sum()
    weights = (1 / frequencies) ** beta  
    return weights 

def median_frequency_weights(n, beta=1):
    """
    Calculate median frequency weights for a list of class counts.
    Args:
        n (list or array-like): List of class counts.
        beta (float): Exponent to adjust the weights. Default is 1.
    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    arr = torch.tensor(n, dtype=torch.float32)
    frequencies = arr / arr.sum()
    median_freq = torch.median(frequencies)  # Get the median of the frequencies
    weights = (median_freq / frequencies) ** beta  # Median frequency divided by each frequency
    return weights 

def normalize(tensor):
    return tensor / tensor.mean()  # Normalize by the mean
    

def log_normalize(tensor):
    t = torch.log(tensor + 1e-10)  # Add a small constant to avoid log(0)
    return t / t.mean()  # Normalize by the mean

def norm_ifw(n, beta=1):
    return normalize(inverse_frequency_weights(n, beta))

def norm_mfw(n, beta=1):
    return normalize(median_frequency_weights(n, beta))

def log_norm_ifw(n, beta=1):
    return log_normalize(inverse_frequency_weights(n, beta))

def log_norm_mfw(n, beta=1):
    return log_normalize(median_frequency_weights(n, beta))



def enet_weights(n, c=1.02):
    """
    Compute ENet class weights from pixel counts.

    w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 
    p_class = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Args:
        n (list or array-like): List of class counts (pixels per class).
        c (float): ENet hyperparameter. Default is 1.02.

    Returns:
        torch.Tensor: ENet class weights as a tensor.
    """
    arr = torch.tensor(n, dtype=torch.float32)
    total = arr.sum()
    propensity_score = arr / total
    weights = 1.0 / torch.log(c + propensity_score)
    return weights

def norm_enetw(n, c=1.02):
    return normalize(enet_weights(n, c))    