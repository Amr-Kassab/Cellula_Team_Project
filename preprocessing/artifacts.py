"""
Artifact Detection and Quality Control Module for BCI Motor Imagery
Evaluates trials for physiological artifacts, electrode pops, clipping, flatlines,
and statistical variance outliers, providing an audit report for scientific documentation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from config import PreprocessingConfig, DEFAULT_CONFIG


def evaluate_trial_artifacts(
    trial_signal: np.ndarray, 
    trial_name: str = 'trial',
    config: PreprocessingConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Evaluate a single trial (shape: n_channels, n_samples) for artifacts.
    
    Parameters
    ----------
    trial_signal : np.ndarray
        Filtered EEG signal of shape (n_channels, n_samples).
    trial_name : str
        Identifier or filename of the trial.
    config : PreprocessingConfig
        Configuration dataclass with threshold values.
        
    Returns
    -------
    result : dict
        Contains 'is_valid', 'reasons', 'p2p_amplitudes', 'variances', etc.
    """
    n_channels, n_samples = trial_signal.shape
    reasons = []
    
    # 1. Peak-to-peak amplitude per channel
    ptp_per_ch = np.ptp(trial_signal, axis=-1)
    var_per_ch = np.var(trial_signal, axis=-1)
    max_abs_per_ch = np.max(np.abs(trial_signal), axis=-1)
    
    for ch_i, ch_name in enumerate(config.channels):
        # Extreme amplitude check (electrode pop, blink, muscle burst)
        if ptp_per_ch[ch_i] > config.max_peak_to_peak_amp:
            reasons.append(f'extreme_ptp_amplitude_{ch_name}_({ptp_per_ch[ch_i]:.1f}uV>{config.max_peak_to_peak_amp}uV)')
            
        # Flatline / Dead channel check
        if var_per_ch[ch_i] < config.min_channel_variance:
            reasons.append(f'flatline_dead_channel_{ch_name}_(var={var_per_ch[ch_i]:.4e})')
            
        # Consecutive constant samples (clipping)
        diffs = np.diff(trial_signal[ch_i])
        zero_diff_count = 0
        max_zero_diff = 0
        for d in diffs:
            if abs(d) < 1e-6:
                zero_diff_count += 1
                if zero_diff_count > max_zero_diff:
                    max_zero_diff = zero_diff_count
            else:
                zero_diff_count = 0
        if max_zero_diff >= config.max_consecutive_flat_samples:
            reasons.append(f'clipping_flat_samples_{ch_name}_({max_zero_diff}_samples)')
            
    is_valid = len(reasons) == 0
    
    return {
        'trial_name': trial_name,
        'is_valid': is_valid,
        'rejection_reasons': '; '.join(reasons) if reasons else 'CLEAN',
        'ptp_per_channel': {ch: float(ptp_per_ch[i]) for i, ch in enumerate(config.channels)},
        'var_per_channel': {ch: float(var_per_ch[i]) for i, ch in enumerate(config.channels)},
        'max_abs_per_channel': {ch: float(max_abs_per_ch[i]) for i, ch in enumerate(config.channels)},
        'overall_var': float(np.mean(var_per_ch)),
        'overall_ptp': float(np.max(ptp_per_ch))
    }


def batch_artifact_rejection(
    signals: np.ndarray, 
    filenames: List[str],
    config: PreprocessingConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Perform comprehensive artifact evaluation and cohort outlier detection across all trials.
    
    Parameters
    ----------
    signals : np.ndarray
        Array of shape (n_trials, n_channels, n_samples).
    filenames : List[str]
        List of filenames corresponding to each trial.
    config : PreprocessingConfig
        Configuration parameters.
        
    Returns
    -------
    valid_mask : np.ndarray
        Boolean array of shape (n_trials,) where True indicates clean trial.
    report_df : pd.DataFrame
        Detailed audit report for every trial.
    """
    n_trials = len(signals)
    trial_reports = []
    
    for i in range(n_trials):
        fname = filenames[i] if i < len(filenames) else f'trial_{i}'
        rep = evaluate_trial_artifacts(signals[i], trial_name=fname, config=config)
        trial_reports.append(rep)
        
    # Statistical cohort variance outlier detection (e.g. trials with abnormal global energy)
    overall_vars = np.array([r['overall_var'] for r in trial_reports])
    # Use robust statistics (median and MAD) to prevent outlier masking
    med_var = np.median(overall_vars)
    mad_var = np.median(np.abs(overall_vars - med_var)) + 1e-9
    robust_z_scores = 0.6745 * (overall_vars - med_var) / mad_var
    
    for i in range(n_trials):
        z = robust_z_scores[i]
        trial_reports[i]['variance_robust_zscore'] = float(z)
        if abs(z) > config.max_zscore_variance:
            if trial_reports[i]['is_valid']:
                trial_reports[i]['is_valid'] = False
                trial_reports[i]['rejection_reasons'] = f'statistical_variance_outlier_(z={z:.2f})'
            else:
                trial_reports[i]['rejection_reasons'] += f'; statistical_variance_outlier_(z={z:.2f})'
                
    valid_mask = np.array([r['is_valid'] for r in trial_reports], dtype=bool)
    
    # Flatten report into DataFrame
    flat_rows = []
    for r in trial_reports:
        row = {
            'filename': r['trial_name'],
            'is_valid': r['is_valid'],
            'rejection_reasons': r['rejection_reasons'],
            'overall_ptp': r['overall_ptp'],
            'overall_var': r['overall_var'],
            'variance_zscore': r.get('variance_robust_zscore', 0.0)
        }
        for ch, ptp in r['ptp_per_channel'].items():
            row[f'ptp_{ch}'] = ptp
        for ch, var in r['var_per_channel'].items():
            row[f'var_{ch}'] = var
        flat_rows.append(row)
        
    report_df = pd.DataFrame(flat_rows)
    return valid_mask, report_df
