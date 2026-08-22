"""
Configuration file for BCI Motor Imagery EEG Preprocessing Pipeline.
Shared across all team members to ensure standard parameters and reproducibility.
"""

from dataclasses import dataclass, field
from typing import List

# ADC to Microvolt conversion for 24-bit OpenBCI / ADS1299 (Gain=24, Vref=4.5V)
ADS1299_SCALE_UV = (4.5 / (24 * (2**23 - 1))) * 1e6  # ~0.02235174 uV/count

@dataclass
class PreprocessingConfig:
    # Sampling parameters
    sampling_rate: float = 250.0  # Uniform target sampling rate in Hz
    num_samples_per_trial: int = 2500  # Raw trial length (10.0 s at 250 Hz)
    
    # Scale conversion
    adc_scale_to_uv: float = ADS1299_SCALE_UV  # Converts raw ADC counts to standard microvolts (uV)
    
    # Channel configuration
    channels: List[str] = field(default_factory=lambda: ['FZ', 'C3', 'CZ', 'C4'])
    
    # Filtering parameters
    bandpass_lowcut: float = 0.5   # Low cut-off frequency in Hz (removes slow drifts and half-cell DC)
    bandpass_highcut: float = 40.0 # High cut-off frequency in Hz (removes high freq EMG noise)
    filter_order: int = 4          # Butterworth filter order (zero-phase sosfiltfilt)
    
    # Motor Imagery specific band (Mu and Beta rhythms: 8 - 30 Hz)
    mu_beta_lowcut: float = 8.0
    mu_beta_highcut: float = 30.0
    
    # Notch filter parameters
    notch_freq: float = 50.0       # Powerline frequency to notch out in Hz (50 Hz or 60 Hz)
    notch_quality_factor: float = 30.0
    apply_notch: bool = True
    
    # Spatial Re-referencing
    reref_method: str = 'CAR'      # 'CAR' (Common Average Reference), 'Laplacian', or 'None'
    
    # Epoching & Baseline Correction parameters
    epoch_start_time: float = 0.5  # Start time of motor imagery epoch window in seconds (after cue onset)
    epoch_end_time: float = 3.5    # End time of motor imagery epoch window in seconds (0.5 - 3.5 s as per PDF)
    baseline_start_time: float = 0.0 # Pre-stimulus baseline start in seconds
    baseline_end_time: float = 0.5   # Pre-stimulus baseline end in seconds
    
    # Artifact Rejection rules (calibrated in microvolts uV)
    max_peak_to_peak_amp: float = 200.0   # Max peak-to-peak amplitude threshold in uV (drops blinks, large EMG)
    min_channel_variance: float = 0.5     # Minimum variance in uV^2 (dead channel/flatline detection)
    max_zscore_variance: float = 4.0      # Outlier trial rejection based on robust variance Z-score across trials
    max_consecutive_flat_samples: int = 15 # Samples with identical values indicating clipping or disconnection
    
    # Outlier clipping / Winsorization (additional recommended step)
    apply_winsorization: bool = True
    winsorize_sigma: float = 4.5          # Soft-clip extreme spikes beyond 4.5 sigma
    
    # Normalization
    normalization_method: str = 'zscore' # 'zscore' (channel-wise standard scaling), 'robust' (IQR), or 'none'
    
    # Random seed
    random_seed: int = 42

DEFAULT_CONFIG = PreprocessingConfig()
