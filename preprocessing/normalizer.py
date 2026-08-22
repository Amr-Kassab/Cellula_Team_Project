"""
Signal Normalization and Outlier Conditioning Module for BCI Motor Imagery
Implements per-channel standardization (Z-score), robust median/IQR scaling,
and statistical winsorization to prevent residual artifact spikes from biasing ML models.
"""

import numpy as np
from typing import Optional
from config import PreprocessingConfig, DEFAULT_CONFIG


def winsorize_channels(
    data: np.ndarray, 
    sigma: float = 4.5
) -> np.ndarray:
    """
    Soft-clip transient amplitude spikes beyond +/- (sigma * standard deviation).
    
    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (..., n_channels, n_samples).
    sigma : float
        Standard deviation multiplier for clipping threshold.
        
    Returns
    -------
    clipped_data : np.ndarray
        Outlier-conditioned signal array.
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True) + 1e-8
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    return np.clip(data, lower_bound, upper_bound)


def normalize_channels(
    data: np.ndarray, 
    method: str = 'zscore',
    apply_winsorize: bool = False,
    winsorize_sigma: float = 4.5,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize EEG channels across the time dimension.
    
    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (..., n_channels, n_samples).
    method : str
        'zscore' : Zero-mean, unit-variance scaling per channel.
        'robust' : Median-centered, IQR scaling per channel.
        'none'   : No scaling applied.
    apply_winsorize : bool
        Whether to soft-clip outliers prior to scaling.
    winsorize_sigma : float
        Sigma threshold for winsorization.
    eps : float
        Small epsilon to prevent division by zero.
        
    Returns
    -------
    normalized_data : np.ndarray
        Normalized signal array with identical shape.
    """
    if apply_winsorize:
        data = winsorize_channels(data, sigma=winsorize_sigma)
        
    method = method.lower()
    
    if method == 'zscore':
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / (std + eps)
        
    elif method == 'robust':
        median = np.median(data, axis=-1, keepdims=True)
        q75 = np.percentile(data, 75, axis=-1, keepdims=True)
        q25 = np.percentile(data, 25, axis=-1, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + eps)
        
    elif method == 'none':
        return data
        
    else:
        raise ValueError(f'Unknown normalization method: {method}')
