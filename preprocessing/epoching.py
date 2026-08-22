"""
Epoching and Baseline Correction Module for BCI Motor Imagery
Extracts task-relevant motor imagery temporal windows and removes pre-stimulus baseline shifts.
"""

import numpy as np
from typing import Tuple, Optional
from config import PreprocessingConfig, DEFAULT_CONFIG


def apply_baseline_correction(
    data: np.ndarray, 
    time_axis: np.ndarray, 
    baseline_start: float = 0.0, 
    baseline_end: float = 0.5
) -> np.ndarray:
    """
    Subtract pre-stimulus baseline mean amplitude from each channel.
    
    Parameters
    ----------
    data : np.ndarray
        EEG signal array of shape (..., n_channels, n_samples).
    time_axis : np.ndarray
        Time points in seconds corresponding to the last axis.
    baseline_start : float
        Start of the pre-stimulus baseline interval in seconds.
    baseline_end : float
        End of the pre-stimulus baseline interval in seconds.
        
    Returns
    -------
    corrected_data : np.ndarray
        Baseline-corrected EEG signal.
    """
    base_mask = (time_axis >= baseline_start) & (time_axis <= baseline_end)
    if not np.any(base_mask):
        # Default to first 10% of samples if window out of range
        n_samples = data.shape[-1]
        base_mask = np.zeros(n_samples, dtype=bool)
        base_mask[:max(1, int(0.1 * n_samples))] = True
        
    baseline_mean = np.mean(data[..., base_mask], axis=-1, keepdims=True)
    return data - baseline_mean


def extract_epoch_window(
    data: np.ndarray, 
    time_axis: np.ndarray, 
    epoch_start: float = 0.5, 
    epoch_end: float = 3.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice the relevant motor imagery task execution epoch window.
    
    Parameters
    ----------
    data : np.ndarray
        EEG signal array of shape (..., n_channels, n_samples).
    time_axis : np.ndarray
        Time points in seconds.
    epoch_start : float
        Epoch start time in seconds (e.g. 0.5s after trial/cue onset).
    epoch_end : float
        Epoch end time in seconds (e.g. 3.5s after trial/cue onset).
        
    Returns
    -------
    epoched_data : np.ndarray
        Sliced EEG signal array of shape (..., n_channels, n_epoch_samples).
    epoch_time_axis : np.ndarray
        Time vector for the sliced epoch window.
    """
    epoch_mask = (time_axis >= epoch_start) & (time_axis <= epoch_end)
    if not np.any(epoch_mask):
        raise ValueError(f'Epoch window [{epoch_start}, {epoch_end}] is outside time axis range [{time_axis[0]}, {time_axis[-1]}]')
        
    epoched_data = data[..., epoch_mask]
    epoch_time_axis = time_axis[epoch_mask]
    return epoched_data, epoch_time_axis
