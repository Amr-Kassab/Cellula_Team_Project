"""
Master EEG Preprocessing Pipeline for BCI Motor Imagery Classification
Encapsulates the end-to-end signal processing stages into a reproducible,
reusable, and deployable pipeline class adhering to scikit-learn conventions.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
import joblib

from config import PreprocessingConfig, DEFAULT_CONFIG
from .filters import apply_eeg_filters
from .re_referencing import apply_rereferencing
from .epoching import apply_baseline_correction, extract_epoch_window
from .artifacts import evaluate_trial_artifacts, batch_artifact_rejection
from .normalizer import normalize_channels

logger = logging.getLogger(__name__)


class EEGPreprocessingPipeline:
    """
    End-to-End EEG Preprocessing Pipeline for Motor Imagery.
    
    Processing Stages:
    1. Physical Scale Calibration (Raw ADC counts -> Microvolts uV)
    2. Uniform Time-Grid Resampling (250 Hz)
    3. DC Offset Suppression & Detrending
    4. Powerline Notch Filtering (50 Hz, Q=30)
    5. Zero-Phase Butterworth Bandpass Filtering (0.5 - 40 Hz)
    6. Spatial Re-referencing (Common Average Reference CAR / Laplacian)
    7. Multi-Criterion Artifact Rejection (PTP threshold, flatline, variance outlier)
    8. Pre-stimulus Baseline Correction (0.0 - 0.5 s)
    9. Motor Imagery Epoch Slicing (0.5 - 3.5 s, 750 samples)
    10. Transient Outlier Conditioning (Winsorization)
    11. Channel-Wise Normalization / Scaling (Z-Score Standardization)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.label_mapping = {'Left': 0, 'Right': 1, 'LEFT': 0, 'RIGHT': 1}
        self.inv_label_mapping = {0: 'Left', 1: 'Right'}
        self.is_fitted = False
        self.audit_log = {}
        
    def transform_single_trial(
        self, 
        signal: np.ndarray, 
        time_axis: Optional[np.ndarray] = None,
        check_artifacts: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process a single EEG trial (channels x samples) through the full pipeline.
        Ideal for real-time inference and web demos (Streamlit / Flask).
        
        Parameters
        ----------
        signal : np.ndarray
            Raw EEG signal of shape (n_channels, n_samples) in uV or raw ADC.
        time_axis : np.ndarray, optional
            Time array in seconds. If None, constructed from config sampling rate.
        check_artifacts : bool
            Whether to run artifact detection checks.
            
        Returns
        -------
        processed_epoch : np.ndarray
            Clean epoched signal of shape (n_channels, n_epoch_samples).
        epoch_time : np.ndarray
            Time vector for the epoch.
        report : dict
            Artifact report and processing metadata.
        """
        n_channels, n_samples = signal.shape
        if time_axis is None:
            time_axis = np.arange(n_samples) / self.config.sampling_rate
            
        # 1. Bandpass & Notch Filtering (with detrending)
        filtered = apply_eeg_filters(signal, self.config)
        
        # 2. Spatial Re-referencing
        reref = apply_rereferencing(filtered, self.config)
        
        # 3. Artifact Evaluation (evaluated on filtered + re-referenced continuous signal)
        if check_artifacts:
            art_report = evaluate_trial_artifacts(reref, trial_name='single_trial', config=self.config)
        else:
            art_report = {'is_valid': True, 'rejection_reasons': 'NOT_CHECKED'}
            
        # 4. Baseline Correction
        base_corrected = apply_baseline_correction(
            reref, 
            time_axis, 
            baseline_start=self.config.baseline_start_time, 
            baseline_end=self.config.baseline_end_time
        )
        
        # 5. Epoch Extraction (Motor Imagery window)
        epoched, epoch_time = extract_epoch_window(
            base_corrected, 
            time_axis, 
            epoch_start=self.config.epoch_start_time, 
            epoch_end=self.config.epoch_end_time
        )
        
        # 6. Normalization with optional winsorization
        normalized = normalize_channels(
            epoched, 
            method=self.config.normalization_method,
            apply_winsorize=self.config.apply_winsorization,
            winsorize_sigma=self.config.winsorize_sigma
        )
        
        return normalized, epoch_time, art_report

    def process_batch(
        self, 
        signals: np.ndarray, 
        time_axis: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of EEG trials through the pipeline with full cohort artifact rejection.
        """
        n_trials, n_channels, n_samples = signals.shape
        if filenames is None:
            filenames = [f'trial_{i+1}.csv' for i in range(n_trials)]
            
        logger.info(f'Starting batch processing of {n_trials} trials...')
        
        # Step 1 & 2: Filtering (Detrending + Notch + Bandpass)
        logger.info(f'Applying Band-pass ({self.config.bandpass_lowcut}-{self.config.bandpass_highcut} Hz) and Notch ({self.config.notch_freq} Hz) filters...')
        filtered = apply_eeg_filters(signals, self.config)
        
        # Step 3: Spatial Re-referencing
        logger.info(f'Applying spatial re-referencing ({self.config.reref_method})...')
        reref = apply_rereferencing(filtered, self.config)
        
        # Step 4: Artifact Rejection Evaluation
        logger.info(f'Running artifact rejection (PTP threshold <= {self.config.max_peak_to_peak_amp} uV)...')
        valid_mask, report_df = batch_artifact_rejection(reref, filenames, self.config)
        
        # Step 5: Baseline Correction
        logger.info(f'Applying pre-stimulus baseline correction [{self.config.baseline_start_time}s - {self.config.baseline_end_time}s]...')
        base_corrected = apply_baseline_correction(
            reref, 
            time_axis, 
            baseline_start=self.config.baseline_start_time, 
            baseline_end=self.config.baseline_end_time
        )
        
        # Step 6: Epoching
        logger.info(f'Extracting epoch window [{self.config.epoch_start_time}s - {self.config.epoch_end_time}s]...')
        epoched, epoch_time = extract_epoch_window(
            base_corrected, 
            time_axis, 
            epoch_start=self.config.epoch_start_time, 
            epoch_end=self.config.epoch_end_time
        )
        
        # Extract raw epoched for comparison
        raw_epoched, _ = extract_epoch_window(
            signals, 
            time_axis, 
            epoch_start=self.config.epoch_start_time, 
            epoch_end=self.config.epoch_end_time
        )
        
        # Step 7: Normalization & Winsorization
        logger.info(f'Applying channel normalization ({self.config.normalization_method}, winsorize={self.config.apply_winsorization})...')
        normalized = normalize_channels(
            epoched, 
            method=self.config.normalization_method,
            apply_winsorize=self.config.apply_winsorization,
            winsorize_sigma=self.config.winsorize_sigma
        )
        
        # Filter clean trials
        clean_X = normalized[valid_mask]
        clean_raw_X = raw_epoched[valid_mask]
        
        clean_y = None
        if labels is not None:
            encoded_y = np.array([self.label_mapping.get(str(lbl).strip(), -1) for lbl in labels])
            clean_y = encoded_y[valid_mask]
            report_df['label'] = labels
            report_df['label_encoded'] = encoded_y
            
        n_clean = int(np.sum(valid_mask))
        n_rejected = int(n_trials - n_clean)
        rejection_rate = (n_rejected / n_trials) * 100.0
        
        logger.info(f'Processing Complete: {n_clean}/{n_trials} trials retained ({n_rejected} rejected, {rejection_rate:.1f}% rejection rate).')
        
        self.audit_log = {
            'total_trials': int(n_trials),
            'clean_trials': int(n_clean),
            'rejected_trials': int(n_rejected),
            'rejection_rate_percent': float(rejection_rate),
            'channel_names': self.config.channels,
            'epoch_duration_sec': float(self.config.epoch_end_time - self.config.epoch_start_time),
            'epoch_n_samples': int(epoched.shape[-1]),
            'sampling_rate': float(self.config.sampling_rate)
        }
        
        return {
            'X_clean': clean_X,
            'y_clean': clean_y,
            'X_raw_epoched': clean_raw_X,
            'valid_mask': valid_mask,
            'clean_indices': np.where(valid_mask)[0],
            'rejected_indices': np.where(~valid_mask)[0],
            'report_df': report_df,
            'epoch_time_axis': epoch_time,
            'audit_log': self.audit_log
        }
        
    def save(self, filepath: str):
        joblib.dump(self, filepath)
        logger.info(f'Pipeline saved to {filepath}')
        
    @staticmethod
    def load(filepath: str) -> 'EEGPreprocessingPipeline':
        return joblib.load(filepath)
