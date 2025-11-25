import torch
import numpy as np


def numpy_to_tensor(data, dtype=torch.float32):
    """
    Convert NumPy array to PyTorch tensor.
    
    Args:
        data (numpy.ndarray): Input NumPy array
        dtype (torch.dtype): Target tensor dtype (default: torch.float32)
    
    Returns:
        torch.Tensor: Converted tensor
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(data)}")
    
    return torch.from_numpy(data).to(dtype)


def numpy_to_tensor_batch(data_dict, dtype=torch.float32):
    """
    Convert multiple NumPy arrays to PyTorch tensors.
    
    Args:
        data_dict (dict): Dictionary mapping names to NumPy arrays
        dtype (torch.dtype): Target tensor dtype (default: torch.float32)
    
    Returns:
        dict: Dictionary mapping names to PyTorch tensors
    """
    return {key: numpy_to_tensor(val, dtype) for key, val in data_dict.items()}


def create_sequences(data, seq_length, target_col_idx=0):
    """
    Create sequences for time series prediction (e.g., for LSTM).
    
    Args:
        data (numpy.ndarray or torch.Tensor): Input data of shape (n_samples, n_features)
        seq_length (int): Length of input sequences
        target_col_idx (int): Index of feature column to use as target (default: 0, typically Close price)
    
    Returns:
        tuple: (X_sequences, y_targets)
            - X_sequences: shape (n_sequences, seq_length, n_features)
            - y_targets: shape (n_sequences, 1) - next value after sequence
    """
    is_tensor = isinstance(data, torch.Tensor)
    
    if is_tensor:
        data_np = data.numpy()
    else:
        data_np = data
    
    X, y = [], []
    
    for i in range(len(data_np) - seq_length):
        X.append(data_np[i:i + seq_length])
        y.append(data_np[i + seq_length, target_col_idx])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    if is_tensor:
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()
    else:
        return X, y


def prepare_for_dataloader(X, y, return_tensors=True):
    """
    Prepare features and targets for PyTorch DataLoader.
    
    Args:
        X (numpy.ndarray): Features array
        y (numpy.ndarray): Targets array
        return_tensors (bool): If True, returns tensors; if False, returns NumPy arrays
    
    Returns:
        tuple: (X, y) as tensors or arrays
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Ensure proper shapes
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    if return_tensors:
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))
        return X_tensor, y_tensor
    else:
        return X.astype(np.float32), y.astype(np.float32)
