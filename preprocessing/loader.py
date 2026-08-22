"""
EEG Data Loader Module for BCI Motor Imagery
Handles ingestion of raw CSV trial files, channel standardization, 
timestamp jitter correction / uniform time grid resampling, ADC to uV conversion, and label mapping.
"""

import os
import re
import glob
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from scipy.interpolate import interp1d

from config import PreprocessingConfig, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_single_trial(
    file_path_or_df, 
    config: PreprocessingConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and standardize a single EEG trial from a CSV file or pandas DataFrame.
    
    Parameters
    ----------
    file_path_or_df : str or pd.DataFrame
        Path to the CSV file or DataFrame containing trial data.
    config : PreprocessingConfig
        Pipeline configuration dataclass.
        
    Returns
    -------
    signal : np.ndarray
        Array of shape (n_channels, n_samples) with standardized channels in microvolts (uV).
    time_axis : np.ndarray
        Uniform time vector in seconds of length n_samples.
    metadata : dict
        Metadata including original sample rate, length, detected drift, etc.
    """
    if isinstance(file_path_or_df, str):
        df = pd.read_csv(file_path_or_df)
    else:
        df = file_path_or_df.copy()
        
    # Standardize column names (strip whitespace, uppercase)
    col_mapping = {col: str(col).strip().upper() for col in df.columns}
    df.rename(columns=col_mapping, inplace=True)
    
    # Verify expected channels
    for ch in config.channels:
        if ch not in df.columns:
            raise ValueError(f'Channel {ch} missing from trial data. Available: {df.columns.tolist()}')
            
    # Extract timestamps and compute raw sampling properties
    has_time = 'TIME' in df.columns
    raw_n_samples = len(df)
    
    if has_time:
        raw_time = df['TIME'].values.astype(np.float64)
        raw_time_rel = raw_time - raw_time[0]
        dt = np.diff(raw_time_rel)
        mean_dt = np.mean(dt) if len(dt) > 0 else (1.0 / config.sampling_rate)
        est_fs = 1.0 / mean_dt if mean_dt > 0 else config.sampling_rate
    else:
        est_fs = config.sampling_rate
        raw_time_rel = np.arange(raw_n_samples) / config.sampling_rate
        
    # Standard target uniform time grid
    target_fs = config.sampling_rate
    target_n_samples = config.num_samples_per_trial
    target_duration = (target_n_samples - 1) / target_fs  # e.g., 9.996s for 2500 samples
    target_time_axis = np.linspace(0.0, target_duration, target_n_samples)
    
    # Extract channel signals and apply microvolt scaling
    raw_signals = df[config.channels].values.T.astype(np.float64) # shape: (n_channels, raw_n_samples)
    if config.adc_scale_to_uv is not None and config.adc_scale_to_uv > 0:
        raw_signals = raw_signals * config.adc_scale_to_uv
    
    # Check if resampling/interpolation to uniform grid is necessary
    if raw_n_samples == target_n_samples and np.allclose(np.diff(raw_time_rel), 1.0/target_fs, rtol=1e-2):
        uniform_signals = raw_signals
    else:
        uniform_signals = np.zeros((len(config.channels), target_n_samples), dtype=np.float64)
        for i in range(len(config.channels)):
            interp_func = interp1d(
                raw_time_rel, 
                raw_signals[i], 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            uniform_signals[i] = interp_func(target_time_axis)
            
    metadata = {
        'original_n_samples': raw_n_samples,
        'resampled_n_samples': target_n_samples,
        'estimated_raw_fs': float(est_fs),
        'target_fs': float(target_fs),
        'channels': config.channels,
        'scaled_to_uv': True
    }
    
    return uniform_signals, target_time_axis, metadata


def load_eeg_dataset(
    data_dir: str = 'raw_data', 
    labels_path: str = 'labels.csv',
    config: PreprocessingConfig = DEFAULT_CONFIG,
    max_trials: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, pd.DataFrame]:
    """
    Load and organize all trial CSV files with their corresponding labels.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'Labels file not found at {labels_path}')
        
    labels_df = pd.read_csv(labels_path)
    label_col = 'label' if 'label' in labels_df.columns else labels_df.columns[0]
    
    # Find all trial files
    pattern = os.path.join(data_dir, 'cellula_MI_data_*.csv')
    file_list = glob.glob(pattern)
    
    if len(file_list) == 0:
        pattern = 'cellula_MI_data_*.csv'
        file_list = glob.glob(pattern)
        
    if len(file_list) == 0 and os.path.exists('cellula_MI_data.csv'):
        file_list = ['cellula_MI_data.csv']
            
    if len(file_list) == 0:
        raise FileNotFoundError(f'No trial files found matching pattern {pattern}')
        
    def extract_index(fpath):
        match = re.search(r'cellula_MI_data_(\d+)\.csv', os.path.basename(fpath))
        if match:
            return int(match.group(1))
        return 0
        
    file_list = sorted(file_list, key=extract_index)
    
    if max_trials is not None:
        file_list = file_list[:max_trials]
        
    logger.info(f'Found {len(file_list)} trial files in {data_dir}. Loading and standardizing...')
    
    signals_list = []
    labels_list = []
    valid_filenames = []
    meta_records = []
    target_time_axis = None
    
    for idx, fpath in enumerate(file_list):
        fname = os.path.basename(fpath)
        file_idx = extract_index(fpath)
        
        # Determine trial label
        if file_idx > 0 and (file_idx - 1) < len(labels_df):
            trial_label = str(labels_df.iloc[file_idx - 1][label_col]).strip()
        elif idx < len(labels_df):
            trial_label = str(labels_df.iloc[idx][label_col]).strip()
        else:
            trial_label = 'Unknown'
            
        try:
            sig, t_axis, meta = load_single_trial(fpath, config)
            if target_time_axis is None:
                target_time_axis = t_axis
                
            signals_list.append(sig)
            labels_list.append(trial_label)
            valid_filenames.append(fname)
            
            meta['trial_id'] = file_idx if file_idx > 0 else (idx + 1)
            meta['filename'] = fname
            meta['label'] = trial_label
            meta_records.append(meta)
            
        except Exception as e:
            logger.warning(f'Failed to load {fname}: {e}')
            
    signals_arr = np.array(signals_list, dtype=np.float64) # (n_trials, n_channels, n_samples)
    labels_arr = np.array(labels_list, dtype=object)
    meta_df = pd.DataFrame(meta_records)
    
    logger.info(
        f'Successfully loaded {len(signals_arr)} trials. '
        f'Shape: {signals_arr.shape} (trials x channels x samples). '
        f'Class distribution: {pd.Series(labels_arr).value_counts().to_dict()}'
    )
    
    return signals_arr, labels_arr, valid_filenames, target_time_axis, meta_df
