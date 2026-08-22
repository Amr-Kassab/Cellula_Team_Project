"""
Main Execution Script for BCI Motor Imagery EEG Preprocessing Pipeline
Runs the complete end-to-end preprocessing workflow on all trial files,
validates quality, logs audit details, and saves processed arrays and figures.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

from config import PreprocessingConfig, DEFAULT_CONFIG
from preprocessing.loader import load_eeg_dataset
from preprocessing.pipeline import EEGPreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def generate_verification_plots(
    raw_signals: np.ndarray,
    clean_signals: np.ndarray,
    raw_epoched: np.ndarray,
    clean_epoched: np.ndarray,
    labels: np.ndarray,
    raw_time: np.ndarray,
    epoch_time: np.ndarray,
    config: PreprocessingConfig,
    report_df: pd.DataFrame,
    output_dir: str = 'figures'
):
    """
    Generate detailed publication-quality verification plots illustrating
    the effects of preprocessing and neurophysiological validity.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.05)
    
    # -------------------------------------------------------------
    # Plot 1: Raw vs Filtered EEG Signals (Time-Domain)
    # -------------------------------------------------------------
    logger.info('Generating Plot 1: Raw vs Filtered Time-Domain comparison...')
    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex='col')
    trial_idx = 0 # Sample clean trial
    
    for ch_i, ch_name in enumerate(config.channels):
        # Raw continuous signal in uV
        raw_trace = raw_signals[trial_idx, ch_i]
        axes[ch_i, 0].plot(raw_time, raw_trace, color='#c0392b', lw=0.9)
        axes[ch_i, 0].set_ylabel(f'{ch_name} (uV)')
        axes[ch_i, 0].set_title(f'{ch_name} - Raw Signal (Drift & Powerline Noise)')
        
        # Clean epoched signal
        axes[ch_i, 1].plot(epoch_time, clean_epoched[trial_idx, ch_i], color='#2980b9', lw=1.1)
        axes[ch_i, 1].set_ylabel(f'{ch_name} (Z-score)')
        axes[ch_i, 1].set_title(f'{ch_name} - Preprocessed Epoch (0.5-3.5s, CAR, 0.5-40Hz)')
        
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.suptitle('EEG Preprocessing Verification: Raw Continuous vs Clean Epoched Trial', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_raw_vs_filtered_time_domain.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: Power Spectral Density (PSD) - Filter Verification
    # -------------------------------------------------------------
    logger.info('Generating Plot 2: PSD Filter Verification...')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    c3_idx = config.channels.index('C3')
    n_sample_trials = min(100, len(clean_epoched))
    sample_raw = raw_signals[:n_sample_trials, c3_idx, :] - np.mean(raw_signals[:n_sample_trials, c3_idx, :], axis=-1, keepdims=True)
    f_raw, psd_raw = signal.welch(sample_raw, fs=config.sampling_rate, nperseg=512, axis=-1)
    mean_psd_raw = np.mean(psd_raw, axis=0)
    
    sample_clean = clean_epoched[:n_sample_trials, c3_idx, :]
    f_clean, psd_clean = signal.welch(sample_clean, fs=config.sampling_rate, nperseg=min(256, clean_epoched.shape[-1]), axis=-1)
    mean_psd_clean = np.mean(psd_clean, axis=0)
    
    ax.semilogy(f_raw, mean_psd_raw, color='#e74c3c', lw=1.8, label='Raw Signal (50 Hz Line Noise & Drift)')
    ax.semilogy(f_clean, mean_psd_clean, color='#27ae60', lw=2.2, label='Preprocessed Signal (0.5-40 Hz Bandpass + 50 Hz Notch + CAR)')
    
    ax.axvspan(8, 12, color='#f39c12', alpha=0.2, label='Mu Band (8-12 Hz)')
    ax.axvspan(13, 30, color='#9b59b6', alpha=0.15, label='Beta Band (13-30 Hz)')
    ax.axvline(50, color='red', linestyle='--', alpha=0.7, label='50 Hz Powerline Notch')
    
    ax.set_xlim(0, 70)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (dB / ^2/Hz$)')
    ax.set_title('Power Spectral Density (PSD) on Channel C3 Before and After Filtering', fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_psd_filtering_verification.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Physiological Target - Left vs Right Motor Imagery PSD on C3 and C4
    # -------------------------------------------------------------
    logger.info('Generating Plot 3: Left vs Right Motor Imagery Spectral Differences...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    c3_idx = config.channels.index('C3')
    c4_idx = config.channels.index('C4')
    
    left_mask = (labels == 0)
    right_mask = (labels == 1)
    
    f_c3_l, psd_c3_left = signal.welch(clean_epoched[left_mask, c3_idx, :], fs=config.sampling_rate, nperseg=256, axis=-1)
    f_c3_r, psd_c3_right = signal.welch(clean_epoched[right_mask, c3_idx, :], fs=config.sampling_rate, nperseg=256, axis=-1)
    
    ax1.plot(f_c3_l, np.mean(psd_c3_left, axis=0), color='#2980b9', lw=2.2, label='Left Hand Imagery')
    ax1.plot(f_c3_r, np.mean(psd_c3_right, axis=0), color='#e67e22', lw=2.2, label='Right Hand Imagery (Contralateral ERD)')
    ax1.axvspan(8, 12, color='#f1c40f', alpha=0.2, label='Mu (8-12 Hz)')
    ax1.axvspan(13, 30, color='#9b59b6', alpha=0.15, label='Beta (13-30 Hz)')
    ax1.set_xlim(4, 40)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title('C3 Electrode (Left Hemisphere / Contralateral to Right Hand)', fontweight='bold')
    ax1.legend()
    
    f_c4_l, psd_c4_left = signal.welch(clean_epoched[left_mask, c4_idx, :], fs=config.sampling_rate, nperseg=256, axis=-1)
    f_c4_r, psd_c4_right = signal.welch(clean_epoched[right_mask, c4_idx, :], fs=config.sampling_rate, nperseg=256, axis=-1)
    
    ax2.plot(f_c4_l, np.mean(psd_c4_left, axis=0), color='#2980b9', lw=2.2, label='Left Hand Imagery (Contralateral ERD)')
    ax2.plot(f_c4_r, np.mean(psd_c4_right, axis=0), color='#e67e22', lw=2.2, label='Right Hand Imagery')
    ax2.axvspan(8, 12, color='#f1c40f', alpha=0.2, label='Mu (8-12 Hz)')
    ax2.axvspan(13, 30, color='#9b59b6', alpha=0.15, label='Beta (13-30 Hz)')
    ax2.set_xlim(4, 40)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('C4 Electrode (Right Hemisphere / Contralateral to Left Hand)', fontweight='bold')
    ax2.legend()
    
    plt.suptitle('Event-Related Desynchronization (ERD) in Mu & Beta Rhythms on C3/C4', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_erd_left_vs_right_psd.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 4: Artifact Rejection Summary & Quality Audit
    # -------------------------------------------------------------
    logger.info('Generating Plot 4: Quality & Artifact Rejection Summary...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    status_counts = report_df['is_valid'].value_counts()
    status_labels = ['Clean / Retained' if v else 'Rejected (Artifacts)' for v in status_counts.index]
    colors = ['#2ecc71', '#e74c3c'] if status_counts.index[0] else ['#e74c3c', '#2ecc71']
    ax1.pie(status_counts, labels=status_labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=[0, 0.08] if len(status_counts)>1 else None)
    ax1.set_title('Trial Acceptance vs Rejection Breakdown', fontweight='bold')
    
    for ch in config.channels:
        sns.kdeplot(np.log10(report_df[f'ptp_{ch}'] + 1e-3), label=f'{ch} PTP', ax=ax2, lw=2)
    ax2.axvline(np.log10(config.max_peak_to_peak_amp), color='red', linestyle='--', label=f'Threshold ({config.max_peak_to_peak_amp} uV)')
    ax2.set_xlabel('log10(Peak-to-Peak Amplitude in uV)')
    ax2.set_ylabel('Density')
    ax2.set_title('Peak-to-Peak Amplitude Distribution Across Channels', fontweight='bold')
    ax2.legend()
    
    plt.suptitle('Data Quality & Artifact Audit Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_artifact_audit_distribution.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 5: Spatial Re-referencing Cross-Channel Correlation
    # -------------------------------------------------------------
    logger.info('Generating Plot 5: Cross-Channel Correlation...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Correlation before CAR
    raw_flat = raw_epoched[:100].transpose(1, 0, 2).reshape(len(config.channels), -1)
    corr_raw = np.corrcoef(raw_flat)
    sns.heatmap(corr_raw, xticklabels=config.channels, yticklabels=config.channels, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1)
    ax1.set_title('Channel Correlation (Before Spatial Filtering)', fontweight='bold')
    
    # Correlation after CAR
    clean_flat = clean_epoched[:100].transpose(1, 0, 2).reshape(len(config.channels), -1)
    corr_clean = np.corrcoef(clean_flat)
    sns.heatmap(corr_clean, xticklabels=config.channels, yticklabels=config.channels, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('Channel Correlation (After CAR & Preprocessing)', fontweight='bold')
    
    plt.suptitle('Spatial Contrast Enhancement via Common Average Reference (CAR)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_channel_cross_correlation.png'), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 6: Time-Frequency Spectrogram (Mu/Beta ERD Dynamics)
    # -------------------------------------------------------------
    logger.info('Generating Plot 6: Time-Frequency Spectrograms...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    
    # Compute spectrogram for Left Imagery on C4 vs Right Imagery on C3
    f_spec, t_spec, s_c3_r = signal.spectrogram(
        clean_epoched[right_mask, c3_idx, :].mean(axis=0), 
        fs=config.sampling_rate, 
        nperseg=64, 
        noverlap=56
    )
    _, _, s_c4_l = signal.spectrogram(
        clean_epoched[left_mask, c4_idx, :].mean(axis=0), 
        fs=config.sampling_rate, 
        nperseg=64, 
        noverlap=56
    )
    
    freq_mask = (f_spec >= 4) & (f_spec <= 35)
    t_spec_adj = t_spec + config.epoch_start_time
    
    im1 = ax1.pcolormesh(t_spec_adj, f_spec[freq_mask], 10*np.log10(s_c3_r[freq_mask, :] + 1e-6), shading='gouraud', cmap='viridis')
    ax1.set_title('C3 Spectrogram - Right Hand Imagery (Contralateral)', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    im2 = ax2.pcolormesh(t_spec_adj, f_spec[freq_mask], 10*np.log10(s_c4_l[freq_mask, :] + 1e-6), shading='gouraud', cmap='viridis')
    ax2.set_title('C4 Spectrogram - Left Hand Imagery (Contralateral)', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    plt.colorbar(im2, ax=ax2, label='Power (dB)')
    
    plt.suptitle('Time-Frequency Dynamic Representation of Motor Imagery ERD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_time_frequency_spectrogram_erd.png'), dpi=200)
    plt.close()
    
    logger.info(f'All verification plots saved to {output_dir}/')


def main():
    logger.info('=== BCI Motor Imagery EEG Preprocessing Pipeline ===')
    
    config = DEFAULT_CONFIG
    logger.info(f'Configuration: Sampling Rate={config.sampling_rate} Hz, Channels={config.channels}')
    logger.info(f'Scale Factor: {config.adc_scale_to_uv:.8f} uV/count')
    logger.info(f'Bandpass: {config.bandpass_lowcut}-{config.bandpass_highcut} Hz, Notch: {config.notch_freq} Hz')
    logger.info(f'Epoch window: [{config.epoch_start_time}s - {config.epoch_end_time}s], Re-ref: {config.reref_method}')
    logger.info(f'Artifact rejection threshold: {config.max_peak_to_peak_amp} uV')
    
    # Step 1: Load and organize dataset
    signals, labels, filenames, time_axis, meta_df = load_eeg_dataset(
        data_dir='raw_data',
        labels_path='labels.csv',
        config=config
    )
    
    # Step 2 to 8: Run end-to-end preprocessing pipeline
    pipeline = EEGPreprocessingPipeline(config=config)
    results = pipeline.process_batch(
        signals=signals,
        time_axis=time_axis,
        labels=labels,
        filenames=filenames
    )
    
    X_clean = results['X_clean']
    y_clean = results['y_clean']
    X_raw_ep = results['X_raw_epoched']
    valid_mask = results['valid_mask']
    report_df = results['report_df']
    epoch_time = results['epoch_time_axis']
    audit_log = results['audit_log']
    
    # Output directory
    out_dir = 'processed_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save processed numpy arrays (.npy)
    logger.info('Saving processed clean dataset arrays...')
    np.save(os.path.join(out_dir, 'X_clean.npy'), X_clean)
    from preprocessing.filters import butter_bandpass_filter
    X_clean_mubeta = butter_bandpass_filter(X_clean, config.mu_beta_lowcut, config.mu_beta_highcut, fs=config.sampling_rate, order=config.filter_order)
    np.save(os.path.join(out_dir, 'X_clean_mubeta.npy'), X_clean_mubeta)
    np.save(os.path.join(out_dir, 'y_clean.npy'), y_clean)
    np.save(os.path.join(out_dir, 'X_raw_epoched.npy'), X_raw_ep)
    np.save(os.path.join(out_dir, 'epoch_time_axis.npy'), epoch_time)
    
    # Save full metadata and artifact audit reports
    report_df.to_csv(os.path.join(out_dir, 'trial_audit_metadata.csv'), index=False)
    
    # Summary of rejection reasons
    rejection_counts = report_df['rejection_reasons'].value_counts().head(20).to_dict()
    rejection_summary = {
        'total_trials': int(len(report_df)),
        'clean_trials': int(np.sum(valid_mask)),
        'rejected_trials': int(np.sum(~valid_mask)),
        'rejection_rate_percent': float(np.mean(~valid_mask) * 100),
        'class_balance_clean': {
            'Left (0)': int(np.sum(y_clean == 0)),
            'Right (1)': int(np.sum(y_clean == 1))
        },
        'clean_trial_shape': list(X_clean.shape),
        'channels': config.channels,
        'sampling_rate_hz': config.sampling_rate,
        'epoch_duration_seconds': float(config.epoch_end_time - config.epoch_start_time),
        'epoch_samples_count': int(X_clean.shape[-1]),
        'top_rejection_reasons_sample': rejection_counts
    }
    
    with open(os.path.join(out_dir, 'rejection_summary.json'), 'w') as f:
        json.dump(rejection_summary, f, indent=4)
        
    # Save config
    config_dict = {
        'sampling_rate': config.sampling_rate,
        'num_samples_per_trial': config.num_samples_per_trial,
        'adc_scale_to_uv': config.adc_scale_to_uv,
        'channels': config.channels,
        'bandpass_lowcut': config.bandpass_lowcut,
        'bandpass_highcut': config.bandpass_highcut,
        'filter_order': config.filter_order,
        'mu_beta_lowcut': config.mu_beta_lowcut,
        'mu_beta_highcut': config.mu_beta_highcut,
        'notch_freq': config.notch_freq,
        'notch_quality_factor': config.notch_quality_factor,
        'reref_method': config.reref_method,
        'epoch_start_time': config.epoch_start_time,
        'epoch_end_time': config.epoch_end_time,
        'baseline_start_time': config.baseline_start_time,
        'baseline_end_time': config.baseline_end_time,
        'max_peak_to_peak_amp': config.max_peak_to_peak_amp,
        'apply_winsorization': config.apply_winsorization,
        'winsorize_sigma': config.winsorize_sigma,
        'normalization_method': config.normalization_method
    }
    with open(os.path.join(out_dir, 'pipeline_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    # Save trained pipeline for deployment / inference
    pipeline.save(os.path.join(out_dir, 'eeg_preprocessing_pipeline.joblib'))
    
    # Generate verification figures
    generate_verification_plots(
        raw_signals=signals[valid_mask],
        clean_signals=signals[valid_mask],
        raw_epoched=X_raw_ep,
        clean_epoched=X_clean,
        labels=y_clean,
        raw_time=time_axis,
        epoch_time=epoch_time,
        config=config,
        report_df=report_df,
        output_dir='figures'
    )
    
    logger.info('=== Preprocessing Pipeline Execution Complete Successfully! ===')
    logger.info(f'Output files in {out_dir}:')
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        logger.info(f' - {f} ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
