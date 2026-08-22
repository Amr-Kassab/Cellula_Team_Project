"""
EEG Filtering Module for BCI Motor Imagery Pipeline
Implements zero-phase Butterworth bandpass filters, notch filters for line noise,
and detrending routines to isolate mu (8-12 Hz) and beta (13-30 Hz) rhythms.
"""

import numpy as np
from scipy import signal
from typing import Union, Tuple, Optional
from config import PreprocessingConfig, DEFAULT_CONFIG


def butter_bandpass_filter(
    data: np.ndarray, 
    lowcut: float, 
    highcut: float, 
    fs: float, 
    order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase forward-backward Butterworth bandpass filter using Second-Order Sections (SOS).
    
    Parameters
    ----------
    data : np.ndarray
        EEG signal array with shape (..., n_samples).
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter.
        
    Returns
    -------
    filtered_data : np.ndarray
        Filtered EEG signal with identical shape and zero phase distortion.
    """
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    
    if low >= high:
        raise ValueError(f'Invalid bandpass frequencies: lowcut={lowcut}, highcut={highcut} for fs={fs}')
        
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return filtered


def notch_filter(
    data: np.ndarray, 
    notch_freq: float, 
    fs: float, 
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply an IIR notch filter to remove power-line interference at a specific frequency (50 Hz or 60 Hz).
    
    Parameters
    ----------
    data : np.ndarray
        EEG signal array with shape (..., n_samples).
    notch_freq : float
        Notch frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    quality_factor : float
        Quality factor Q = w0 / bw characterizing the notch bandwidth.
        
    Returns
    -------
    notched_data : np.ndarray
        Filtered EEG signal with notch frequency attenuated.
    """
    nyq = 0.5 * fs
    if notch_freq >= nyq:
        # Notch frequency exceeds Nyquist, no filtering needed
        return data
        
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    notched = signal.filtfilt(b, a, data, axis=-1)
    return notched


def apply_eeg_filters(
    data: np.ndarray, 
    config: PreprocessingConfig = DEFAULT_CONFIG,
    mu_beta_only: bool = False
) -> np.ndarray:
    """
    Complete filtering pipeline for EEG signals:
    1. DC offset removal / linear detrending.
    2. Power-line notch filter (e.g., 50 Hz).
    3. Zero-phase Butterworth bandpass filter (0.5-40 Hz or 8-30 Hz).
    
    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (n_channels, n_samples) or (n_trials, n_channels, n_samples).
    config : PreprocessingConfig
        Pipeline configuration settings.
    mu_beta_only : bool
        If True, applies tighter 8-30 Hz bandpass filter for sensorimotor rhythms.
        
    Returns
    -------
    clean_filtered : np.ndarray
        Zero-phase filtered signals.
    """
    # 1. Detrend to remove massive DC offset and baseline slope
    detrended = signal.detrend(data, axis=-1, type='linear')
    
    # 2. Notch filter if enabled
    if config.apply_notch and config.notch_freq < (0.5 * config.sampling_rate):
        filtered = notch_filter(detrended, config.notch_freq, config.sampling_rate, config.notch_quality_factor)
    else:
        filtered = detrended
        
    # 3. Bandpass filter
    if mu_beta_only:
        lowcut = config.mu_beta_lowcut
        highcut = config.mu_beta_highcut
    else:
        lowcut = config.bandpass_lowcut
        highcut = config.bandpass_highcut
        
    filtered = butter_bandpass_filter(filtered, lowcut, highcut, config.sampling_rate, config.filter_order)
    return filtered
