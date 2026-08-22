"""
BCI Motor Imagery EEG Preprocessing Pipeline Package
"""

from .loader import load_eeg_dataset, load_single_trial
from .filters import butter_bandpass_filter, notch_filter, apply_eeg_filters
from .re_referencing import apply_car, apply_laplacian_c3_c4
from .epoching import apply_baseline_correction, extract_epoch_window
from .artifacts import evaluate_trial_artifacts, batch_artifact_rejection
from .normalizer import normalize_channels
from .pipeline import EEGPreprocessingPipeline

__all__ = [
    'load_eeg_dataset',
    'load_single_trial',
    'butter_bandpass_filter',
    'notch_filter',
    'apply_eeg_filters',
    'apply_car',
    'apply_laplacian_c3_c4',
    'apply_baseline_correction',
    'extract_epoch_window',
    'evaluate_trial_artifacts',
    'batch_artifact_rejection',
    'normalize_channels',
    'EEGPreprocessingPipeline'
]
