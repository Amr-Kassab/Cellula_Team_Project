"""
Spatial Re-referencing Module for BCI Motor Imagery EEG
Implements Common Average Reference (CAR) and Local Laplacian / Bipolar derivations
to eliminate global environmental noise and enhance local sensorimotor cortex signal contrast.
"""

import numpy as np
from typing import List, Optional
from config import PreprocessingConfig, DEFAULT_CONFIG


def apply_car(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) across channels.
    
    V_i_CAR(t) = V_i(t) - (1/N_ch) * sum_k(V_k(t))
    
    Parameters
    ----------
    data : np.ndarray
        EEG signal of shape (n_channels, n_samples) or (n_trials, n_channels, n_samples).
        The channel dimension is always the second to last axis (-2).
        
    Returns
    -------
    car_data : np.ndarray
        Common average re-referenced signal.
    """
    mean_across_channels = np.mean(data, axis=-2, keepdims=True)
    return data - mean_across_channels


def apply_laplacian_c3_c4(
    data: np.ndarray, 
    channel_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Apply a localized Laplacian / differential spatial filter targeting C3 and C4 channels.
    Specifically computes local spatial contrast relative to midline (CZ, FZ).
    
    Parameters
    ----------
    data : np.ndarray
        EEG signal of shape (n_channels, n_samples) or (n_trials, n_channels, n_samples).
    channel_names : List[str], optional
        List of channel names in the order they appear in data. Defaults to ['FZ', 'C3', 'CZ', 'C4'].
        
    Returns
    -------
    laplacian_data : np.ndarray
        Re-referenced signal enhancing sensorimotor activation.
    """
    if channel_names is None:
        channel_names = ['FZ', 'C3', 'CZ', 'C4']
        
    ch_idx = {name: i for i, name in enumerate(channel_names)}
    
    out = data.copy()
    has_fz = 'FZ' in ch_idx
    has_c3 = 'C3' in ch_idx
    has_cz = 'CZ' in ch_idx
    has_c4 = 'C4' in ch_idx
    
    if has_c3 and has_cz:
        c3_i = ch_idx['C3']
        cz_i = ch_idx['CZ']
        if has_fz:
            fz_i = ch_idx['FZ']
            # C3 local laplacian: C3 - 0.5 * (CZ + FZ)
            out[..., c3_i, :] = data[..., c3_i, :] - 0.5 * (data[..., cz_i, :] + data[..., fz_i, :])
        else:
            out[..., c3_i, :] = data[..., c3_i, :] - data[..., cz_i, :]
            
    if has_c4 and has_cz:
        c4_i = ch_idx['C4']
        cz_i = ch_idx['CZ']
        if has_fz:
            fz_i = ch_idx['FZ']
            # C4 local laplacian: C4 - 0.5 * (CZ + FZ)
            out[..., c4_i, :] = data[..., c4_i, :] - 0.5 * (data[..., cz_i, :] + data[..., fz_i, :])
        else:
            out[..., c4_i, :] = data[..., c4_i, :] - data[..., cz_i, :]
            
    return out


def apply_rereferencing(
    data: np.ndarray, 
    config: PreprocessingConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """
    Apply spatial re-referencing according to configuration.
    """
    method = config.reref_method.upper()
    if method == 'CAR':
        return apply_car(data)
    elif method == 'LAPLACIAN':
        return apply_laplacian_c3_c4(data, config.channels)
    elif method == 'NONE':
        return data
    else:
        raise ValueError(f'Unknown re-referencing method: {config.reref_method}')
