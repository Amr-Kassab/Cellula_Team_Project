# BCI Motor Imagery Classification — EEG Preprocessing & Quality Engineering Report

**Project Role**: Data & Preprocessing Pipeline Lead  
**Target Physiological Mechanism**: Sensorimotor Cortex Event-Related Desynchronization (ERD/ERS) in mu (8–12 Hz) and beta (13–30 Hz) rhythms over C3 and C4  
**Date**: August 2026  

---

## 1. Executive Summary & Deliverables

This deliverable provides the **clean, standardized, artifact-filtered, and calibrated EEG dataset** along with a production-ready, modular preprocessing pipeline for the team's machine learning (CSP + SVM) and deep learning (EEGNet / CNN-LSTM) workflows.

### Summary of Dataset Deliverables in `processed_data/`:
| File Name | Dimensions / Size | Description | Intended Use |
| :--- | :--- | :--- | :--- |
| **`X_clean.npy`** | `(1031, 4, 751)` | Clean epoched EEG signals (0.5-40 Hz, CAR, Z-score normalized) | Deep Learning models (EEGNet, CNN, ConvNet) |
| **`X_clean_mubeta.npy`** | `(1031, 4, 751)` | Clean epoched EEG signals bandpassed in mu/beta band (8-30 Hz) | Classical ML (CSP + SVM / LDA, Bandpower) |
| **`y_clean.npy`** | `(1031,)` | Encoded labels (`0` = Left, `1` = Right) | Ground truth target vector |
| **`X_raw_epoched.npy`** | `(1031, 4, 751)` | Unprocessed epoched baseline signals | Baseline comparisons & ablation experiments |
| **`epoch_time_axis.npy`**| `(751,)` | Relative time vector from 0.5 s to 3.5 s | Plotting & time-frequency analysis |
| **`trial_audit_metadata.csv`** | `2,160` rows | Complete audit log of every trial with PTP amplitudes, variances, Z-scores, and rejection reasons | Scientific reproducibility & documentation |
| **`rejection_summary.json`** | JSON report | Statistical breakdown of rejection causes and class distribution | Quality assurance tracking |
| **`pipeline_config.json`** | JSON config | Full parameters used in the pipeline | Shared team configuration |
| **`eeg_preprocessing_pipeline.joblib`** | Serialized object | Reusable pipeline class instance | Streamlit / Flask inference web app |

---

## 2. Preprocessing Architecture & Ordered Steps

Following the specifications in Table 1 of the project document and best practices in neuroengineering, the pipeline executes the following 8 ordered stages:

```
Raw Multi-Channel CSVs (2160 Trials)
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. Load & Calibrate                                    │
 │    • Convert OpenBCI 24-bit ADC counts to µV (0.02235) │
 │    • Uniform time-grid interpolation (Fs = 250.0 Hz)   │
 │    • Standardize channels: [FZ, C3, CZ, C4]            │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. DC Offset Removal & Powerline Notch Filtering       │
 │    • Linear detrending (suppresses half-cell DC bias)  │
 │    • Zero-phase 50 Hz IIR Notch filter (Q = 30.0)      │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Zero-Phase Butterworth Bandpass Filtering           │
 │    • 4th-order SOS filter (0.5 – 40.0 Hz)              │
 │    • Preserves exact phase timing for ERD detection    │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Spatial Re-referencing                              │
 │    • Common Average Reference (CAR): V_i - mean(V)     │
 │    • Suppresses global shared noise across electrodes  │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 5. Multi-Criterion Artifact Rejection & Audit          │
 │    • Peak-to-Peak Amplitude threshold: PTP <= 200 µV   │
 │    • Dead channel / flatline detection: var >= 0.5 µV² │
 │    • Robust cohort variance outlier check: |Z| <= 4.0  │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 6. Pre-Stimulus Baseline Correction                    │
 │    • Baseline interval: 0.0 s – 0.5 s                  │
 │    • Subtract channel-wise baseline mean               │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 7. Motor Imagery Task Epoch Slicing                    │
 │    • Cue-locked temporal window: 0.5 s – 3.5 s (751 pts)│
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ 8. Conditioning & Per-Channel Normalization            │
 │    • Soft-clipping of residual spikes (Winsorize 4.5σ) │
 │    • Channel-wise Z-Score Standardization (µ=0, σ=1)   │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
         Final Clean Arrays & Pipeline Model
```

---

## 3. Additional Recommended Steps Implemented

Beyond the minimum specifications in the project PDF, the following enhancements were implemented:

1. **Hardware-Accurate Microvolt Calibration**:
   The raw files contained raw 24-bit integer ADC counts from an ADS1299 amplifier (common in OpenBCI Cyton boards). Applying the exact scaling constant:
   $$\text{Scale Factor} = \frac{4.5\text{ V}}{24 \times (2^{23} - 1)} \times 10^6 = 0.02235174\ \mu\text{V/count}$$
   restored physiologically meaningful amplitudes ($20-100\ \mu\text{V}$).

2. **Jitter Correction & Resampling onto Uniform Time Grid**:
   Real-time hardware streaming often suffers from slight inter-sample timestamp jitter. Each trial was interpolated onto an exact uniform $250.0\text{ Hz}$ time grid ($2,500$ raw samples for $10.0\text{ s}$ duration), preventing spectral distortion.

3. **Dual Array Output for Modeling Flexibility**:
   - **`X_clean.npy` (0.5 – 40.0 Hz)**: Preserves the entire broadband spectrum so Deep Learning models (EEGNet) can optimize their own spatial and temporal convolution kernels.
   - **`X_clean_mubeta.npy` (8.0 – 30.0 Hz)**: Specifically band-limited to mu and beta sensorimotor rhythms for optimal performance in Common Spatial Pattern (CSP) feature extraction.

4. **Statistical Winsorization ($4.5\sigma$)**:
   Clean trials may occasionally contain brief single-sample contact glitches. Soft-clipping values beyond $4.5\sigma$ prevents outlier bias without distorting underlying sinusoidal rhythm dynamics.

5. **Cross-Channel Correlation Decoupling**:
   Before CAR, all channels exhibited $0.93 - 0.98$ cross-correlation due to shared environmental noise. After CAR, cross-correlation dropped to $-0.27$, unmasking localized cortical contrast between C3 and C4.

6. **Streamlit/Inference Pipeline Serializer**:
   The complete pipeline is packaged in `eeg_preprocessing_pipeline.joblib`, allowing single-trial inference for the web deployment team member with a single function call:
   ```python
   clean_epoch, time_axis, report = pipeline.transform_single_trial(raw_trial_csv_or_df)
   ```

---

## 4. Quality Audit & Rejection Statistics

- **Total Trials Processed**: 2,160
- **Clean Trials Retained**: 1,031 (47.7%)
- **Rejected Trials**: 1,129 (52.3%)
- **Class Balance of Clean Trials**:
  - **Left Hand (Class 0)**: 533 trials (51.7%)
  - **Right Hand (Class 1)**: 498 trials (48.3%)
  - *Result*: Near-perfect 50/50 balance, preventing classifier bias.

### Primary Causes of Trial Rejections:
1. **Severe Motion / Muscle Bursts (EMG)**: Peak-to-peak amplitude exceeding $200\ \mu\text{V}$ (often thousands of $\mu\text{V}$).
2. **Electrode Pops & Baseline Jumps**: Rapid voltage steps causing high statistical variance ($Z > 4.0$).
3. **Dead / Disconnected Channels**: Channels with flatlined ADC values or near-zero variance.

Every rejected trial and its explicit reason is logged in `processed_data/trial_audit_metadata.csv`.

---

## 5. Verification Figures

All generated verification figures are stored in the `figures/` directory:
- **`01_raw_vs_filtered_time_domain.png`**: Multi-channel raw vs cleaned time-domain traces.
- **`02_psd_filtering_verification.png`**: Power Spectral Density before and after filtering (verifying 50 Hz notch and 0.5-40 Hz bandpass).
- **`03_erd_left_vs_right_psd.png`**: Event-Related Desynchronization (ERD) spectral comparison on C3 and C4.
- **`04_artifact_audit_distribution.png`**: Rejection pie chart and channel amplitude distributions.
- **`05_channel_cross_correlation.png`**: Channel correlation matrices demonstrating spatial contrast enhancement via CAR.
- **`06_time_frequency_spectrogram_erd.png`**: Spectrograms illustrating time-frequency power changes during motor imagery.

---

## 6. How Team Members Can Load and Use the Clean Data

### For Classical ML (CSP + SVM) Lead:
```python
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load mu/beta filtered clean data
X = np.load('processed_data/X_clean_mubeta.npy')  # shape: (1031, 4, 751)
y = np.load('processed_data/y_clean.npy')         # shape: (1031,)

# Build CSP + SVM pipeline
csp = CSP(n_components=4, log=True, norm_trace=False)
clf = Pipeline([
    ('csp', csp),
    ('svm', SVC(kernel='linear', C=1.0))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f'CSP + SVM 5-Fold Accuracy: {scores.mean()*100:.2f}%')
```

### For Deep Learning (EEGNet) Lead:
```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load broadband clean data
X = np.load('processed_data/X_clean.npy')  # shape: (1031, 4, 751)
y = np.load('processed_data/y_clean.npy')

# Reshape for EEGNet: (batch_size, 1, n_channels, n_samples)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f'EEGNet Input Tensor Shape: {X_tensor.shape}')
```

### For Deployment / Web Demo (Streamlit) Lead:
```python
import joblib
import numpy as np
import pandas as pd

# Load saved preprocessor
preprocessor = joblib.load('processed_data/eeg_preprocessing_pipeline.joblib')

# Ingest uploaded user trial CSV
uploaded_df = pd.read_csv('sample_trial.csv')
clean_epoch, time_axis, report = preprocessor.transform_single_trial(uploaded_df)

if not report['is_valid']:
    print(f'Warning: Trial rejected due to: {report["rejection_reasons"]}')
else:
    # Run trained model inference
    prediction = model.predict(clean_epoch[np.newaxis, ...])
    print(f'Predicted Imagery: {"Right" if prediction[0] == 1 else "Left"}')
```
