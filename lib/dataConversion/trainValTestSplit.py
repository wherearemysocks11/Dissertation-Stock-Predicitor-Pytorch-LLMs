import numpy as np
import torch
from sklearn.model_selection import train_test_split


def split_train_val_test(X, y, train_size=0.95, val_size=0.05, test_size=0.05, 
                          random_state=42, shuffle=True):
    """
    Split data into train, validation, and test sets with randomization.
    Returns NumPy arrays for easier manipulation before converting to tensors.
    
    Note: train_size + val_size + test_size should equal 1.0, but we handle
    the case where they sum to more than 1.0 by adjusting proportions.
    
    Args:
        X (numpy.ndarray): Features array of shape (n_samples, n_features)
        y (numpy.ndarray): Targets array of shape (n_samples, 1) or (n_samples,)
        train_size (float): Proportion for training (default: 0.95)
        val_size (float): Proportion for validation (default: 0.05)
        test_size (float): Proportion for testing (default: 0.05)
        random_state (int): Random seed for reproducibility (default: 42)
        shuffle (bool): Whether to shuffle data before splitting (default: True)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
               All as NumPy arrays for easier manipulation
    """
    # Ensure NumPy arrays
    if not isinstance(X, np.ndarray):
        X_np = np.array(X)
    else:
        X_np = X
    
    if not isinstance(y, np.ndarray):
        y_np = np.array(y)
    else:
        y_np = y
    
    # Normalize proportions if they don't sum to 1.0
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        print(f"Warning: Proportions sum to {total}, normalizing to 1.0")
        train_size = train_size / total
        val_size = val_size / total
        test_size = test_size / total
    
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_np, y_np,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    
    # Second split: split remaining data into train and validation
    # Calculate validation proportion from remaining data
    val_proportion = val_size / (train_size + val_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_proportion,
        random_state=random_state,
        shuffle=shuffle
    )
    
    print(f"\nData split complete:")
    print(f"  Training:   {len(X_train):5d} samples ({len(X_train)/len(X_np)*100:5.1f}%)")
    print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(X_np)*100:5.1f}%)")
    print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(X_np)*100:5.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_sequences_train_val_test(X_seq, y_seq, train_size=0.95, val_size=0.05, 
                                     test_size=0.05, random_state=42, shuffle=True):
    """
    Split sequence data (for LSTM/RNN) into train, validation, and test sets.
    Returns NumPy arrays.
    
    Args:
        X_seq (numpy.ndarray): Sequence features of shape (n_sequences, seq_length, n_features)
        y_seq (numpy.ndarray): Sequence targets of shape (n_sequences, 1) or (n_sequences,)
        train_size (float): Proportion for training (default: 0.95)
        val_size (float): Proportion for validation (default: 0.05)
        test_size (float): Proportion for testing (default: 0.05)
        random_state (int): Random seed for reproducibility (default: 42)
        shuffle (bool): Whether to shuffle sequences before splitting (default: True)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) as NumPy arrays
    """
    # Use the same logic as regular split
    return split_train_val_test(X_seq, y_seq, train_size, val_size, test_size, 
                                 random_state, shuffle)


def stratified_time_split(X, y, dates, train_size=0.95, val_size=0.05, test_size=0.05,
                          random_state=42):
    """
    Split data ensuring representation across all time periods (stratified by month/quarter).
    This helps ensure validation and test sets contain samples from throughout the time series,
    not just recent data. Returns NumPy arrays for easier manipulation.
    
    Args:
        X (numpy.ndarray): Features array
        y (numpy.ndarray): Targets array
        dates (pandas.DatetimeIndex): Corresponding dates for each sample
        train_size (float): Proportion for training (default: 0.95)
        val_size (float): Proportion for validation (default: 0.05)
        test_size (float): Proportion for testing (default: 0.05)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, 
                dates_train, dates_val, dates_test) all as NumPy/pandas
    """
    # Ensure NumPy arrays
    if not isinstance(X, np.ndarray):
        X_np = np.array(X)
    else:
        X_np = X
    
    if not isinstance(y, np.ndarray):
        y_np = np.array(y)
    else:
        y_np = y
    
    # Create stratification labels (by year-month)
    strata = dates.to_period('M').astype(str)
    
    # Normalize proportions
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        train_size = train_size / total
        val_size = val_size / total
        test_size = test_size / total
    
    # First split: separate test set (stratified by month)
    X_temp, X_test, y_temp, y_test, dates_temp, dates_test = train_test_split(
        X_np, y_np, dates,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
        shuffle=True
    )
    
    # Recalculate strata for remaining data
    strata_temp = dates_temp.to_period('M').astype(str)
    
    # Second split: train and validation
    val_proportion = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
        X_temp, y_temp, dates_temp,
        test_size=val_proportion,
        random_state=random_state,
        stratify=strata_temp,
        shuffle=True
    )
    
    print(f"\nStratified time-based split complete:")
    print(f"  Training:   {len(X_train):5d} samples ({len(X_train)/len(X_np)*100:5.1f}%)")
    print(f"    Date range: {dates_train.min()} to {dates_train.max()}")
    print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(X_np)*100:5.1f}%)")
    print(f"    Date range: {dates_val.min()} to {dates_val.max()}")
    print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(X_np)*100:5.1f}%)")
    print(f"    Date range: {dates_test.min()} to {dates_test.max()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test
