# BCI Motor Imagery Classification — Comprehensive Study & Engineering Guide

**Project Title**: Brain-Computer Interface (BCI) Motor Imagery Classification  
**Target Physiological Phenomenon**: Sensorimotor Rhythm (SMR) Modulation — Mu (8–12 Hz) & Beta (13–30 Hz) Event-Related Desynchronization (ERD) over Sensorimotor Cortices (C3, C4)  
**Input Modality**: Multi-Channel Electroencephalography (EEG) — Electrodes: `FZ`, `C3`, `CZ`, `C4`  
**Classification Task**: Binary Classification — Imagined **LEFT Hand** Movement (`Class 0`) vs Imagined **RIGHT Hand** Movement (`Class 1`)  

---

## Table of Contents
1. [Project File Inventory & System Architecture](#1-project-file-inventory--system-architecture)
2. [Section 1: Pre-processing](#2-section-1-pre-processing)
3. [Section 2: Exploratory Data Analysis (EDA)](#3-section-2-exploratory-data-analysis-eda)
4. [Section 3: Models & Empirical Evaluation](#4-section-3-models--empirical-evaluation)
5. [Section 4: Deployment](#5-section-4-deployment)
6. [Section 5: Guide to Improving Results Across Every Section](#6-section-5-guide-to-improving-results-across-every-section)
7. [Section 6: Plain-English Defense & Presentation Guide (Simple Terms for Instructors)](#7-plain-english-defense--presentation-guide-simple-terms-for-instructors)

---

## 1. Project File Inventory & System Architecture

This repository is organized into four distinct engineering subsystems: **Data & Preprocessing**, **Exploratory Data Analysis**, **Machine Learning & Deep Learning Modeling**, and **Web Deployment**. Below is the complete catalog of every file in the project, what it does, and how it works under the hood.

### 1.1 Root Scripts & Configuration
| File | Role | Implementation Mechanics |
| :--- | :--- | :--- |
| **`config.py`** | Central configuration file for all pipeline parameters. | Uses a Python `@dataclass` (`PreprocessingConfig`) to encapsulate sampling rate (250 Hz), trial length (2500 samples), ADC-to-uV scale factor (`0.02235174`), filter cutoffs (0.5–40 Hz broadband, 8–30 Hz mu/beta), notch frequency (50 Hz, Q=30), epoch window (0.5–3.5s), baseline window (0.0–0.5s), artifact rejection thresholds (PTP <= 200 uV, variance >= 0.5 uV², |Z| <= 4.0), and normalization methods. |
| **`run_preprocessing.py`** | Batch preprocessing orchestrator and verification suite. | Ingests all 2,160 CSV files from `raw_data/`, applies `EEGPreprocessingPipeline`, filters artifacts, saves clean arrays (`.npy`), exports CSV audit metadata, and produces 6 publication-grade diagnostic verification plots. |
| **`run_modeling.py`** | Command-line interface for model training and evaluation. | Uses `argparse` to run individual models (`csp_svm`, `eegnet`, `cnn`, `cnn_lstm`, `transformer`) or `all`. Invokes cross-validation, logs out-of-fold metrics, and supports an `--audit` flag for data integrity diagnostics. |
| **`document.pdf`** | Official project specifications and roadmap document. | End-to-end guide detailing dataset parameters, recommended 8-step preprocessing sequence, modeling roadmap, deployment recommendations, and team responsibilities. |
| **`labels.csv`** | Ground truth label file for all trials. | Single-column CSV with 2,400 rows containing string labels (`Left`, `Right`). Rows 0–2159 correspond to files `cellula_MI_data_1.csv` through `cellula_MI_data_2160.csv`. |
| **`cellula_MI_data.csv`** | Representative single trial EEG CSV. | 2,500 rows and 5 columns (`Time`, `FZ`, `C3`, `CZ`, `C4`) representing a 10-second recording sampled at ~250 Hz. Used for standalone testing and deployment demonstration. |
| **`eeg_preprocessing_pipeline.ipynb`** | Interactive Jupyter Notebook for preprocessing. | Self-contained notebook walking through raw data inspection, filtering verification, artifact audit, and array export with step-by-step visualizations. |
| **`PROCESSING_REPORT.md`** | Preprocessing engineering and audit report. | Detailed report on dataset properties, artifact rejection breakdown, class distribution, and usage instructions for team members. |
| **`README.md`** | GitHub repository landing page and documentation. | High-level overview, architecture diagrams, installation instructions, quickstart snippets for ML/DL/Deployment, and result summaries. |

---

### 1.2 Preprocessing Package (`preprocessing/`)
| File | Role | Implementation Mechanics |
| :--- | :--- | :--- |
| **`preprocessing/__init__.py`** | Package namespace export. | Exposes core loader, filter, spatial referencing, epoching, artifact, normalizer, and pipeline classes for clean imports. |
| **`preprocessing/loader.py`** | Data ingestion, calibration, and temporal standardization. | Standardizes column headers, applies ADC-to-uV calibration, calculates sampling frequency from timestamps, and performs 1D linear/spline interpolation (`scipy.interpolate.interp1d`) onto an exact uniform 250 Hz time grid (2,500 samples over 9.996s). Matches file indices to `labels.csv`. |
| **`preprocessing/filters.py`** | Zero-phase digital filtering routines. | Implements zero-phase forward-backward IIR Butterworth bandpass filtering via Second-Order Sections (`scipy.signal.butter(..., output='sos')` and `scipy.signal.sosfiltfilt`) to ensure zero phase distortion. Implements 50 Hz powerline notch filtering (`scipy.signal.iirnotch` and `filtfilt`). Includes linear detrending (`scipy.signal.detrend`) to suppress half-cell DC bias. |
| **`preprocessing/re_referencing.py`** | Spatial filtering and re-referencing. | Implements Common Average Reference (CAR: $V_i - \frac{1}{N}\sum V_k$) to eliminate common-mode scalp noise. Also implements local Laplacian contrast for C3/C4 relative to midline electrodes. |
| **`preprocessing/epoching.py`** | Baseline correction and task window slicing. | Calculates mean amplitude across pre-stimulus interval (0.0–0.5s) per channel and subtracts it from the entire signal. Slices the active motor imagery execution window (0.5–3.5s post-cue, yielding 751 time samples). |
| **`preprocessing/artifacts.py`** | Multi-criterion physiological artifact evaluation. | Evaluates peak-to-peak (PTP) amplitude per channel against a 200 uV threshold, checks for dead channels/flatlines (variance < 0.5 uV² or >=15 identical consecutive samples), and calculates robust cohort variance Z-scores ($0.6745 \times \frac{\text{Var} - \text{Median}}{\text{MAD}}$) to reject statistical energy outliers (|Z| > 4.0). Generates comprehensive trial-by-trial audit records. |
| **`preprocessing/normalizer.py`** | Statistical conditioning and amplitude scaling. | Implements statistical Winsorization (soft-clipping values beyond $\pm 4.5\sigma$) to suppress residual contact glitches. Implements per-channel Z-score standardization ($\frac{x - \mu}{\sigma + \epsilon}$) and robust IQR scaling. |
| **`preprocessing/pipeline.py`** | Master pipeline coordinator and deployable class. | Encapsulates all 8 processing stages into `transform_single_trial()` (for single-trial real-time inference) and `process_batch()` (for cohort processing). Handles label encoding (0: Left, 1: Right), audit compilation, and pipeline serialization with `joblib`. |

---

### 1.3 Preprocessed Deliverables (`processed_data/`)
| File | Dimensions / Size | Purpose & Technical Format |
| :--- | :--- | :--- |
| **`X_clean.npy`** | `(1031, 4, 751)` float64 (24 MB) | Clean epoched signals filtered at 0.5–40 Hz broadband, CAR re-referenced, baseline-corrected, and normalized. Primary input for **EEGNet**, **CNN**, **CNN-LSTM**, and **Transformer**. |
| **`X_clean_mubeta.npy`** | `(1031, 4, 751)` float64 (24 MB) | Clean epoched signals specifically bandpass-filtered to sensorimotor rhythm range (8–30 Hz). Primary input for **CSP + SVM** spatial covariance feature extraction. |
| **`y_clean.npy`** | `(1031,)` int64 (8.2 KB) | Encoded binary ground truth labels corresponding to clean trials (`0` = Left Hand, `1` = Right Hand). |
| **`X_raw_epoched.npy`** | `(1031, 4, 751)` float64 (24 MB) | Epoched signals without bandpass filtering, CAR, or normalization. Retained for ablation studies and baseline benchmarking. |
| **`epoch_time_axis.npy`** | `(751,)` float64 (6.0 KB) | Temporal coordinates in seconds ($t \in [0.5, 3.5]$ s) at 250 Hz sampling rate. |
| **`trial_audit_metadata.csv`**| 2,160 rows, 13 columns (669 KB) | Complete audit table documenting every single raw trial: filename, trial ID, label, acceptance status (`is_valid`), rejection reasons, channel PTP amplitudes, channel variances, and robust Z-score. |
| **`rejection_summary.json`** | JSON dictionary (3.1 KB) | Quantitative quality assurance report detailing total trials (2,160), clean trials (1,031), rejected trials (1,129), rejection rate (52.27%), class balance, and breakdown of rejection causes. |
| **`pipeline_config.json`** | JSON dictionary (664 B) | Machine-readable dump of all configuration hyperparameters used to produce the processed dataset. |
| **`eeg_preprocessing_pipeline.joblib`** | Serialized object (1.1 KB) | Fitted instance of `EEGPreprocessingPipeline` saved via `joblib`, ready for zero-overhead inference in production environments. |

---

### 1.4 Diagnostic Figures (`figures/`)
| File | Graphic Contents & Diagnostic Significance |
| :--- | :--- |
| **`01_raw_vs_filtered_time_domain.png`** | 4-channel side-by-side time-domain comparison showing raw continuous drift and high-frequency noise on FZ, C3, CZ, C4 versus clean epoched signals. |
| **`02_psd_filtering_verification.png`** | Welch Power Spectral Density (PSD) on C3 before vs after filtering. Demonstrates strong attenuation of 50 Hz powerline hum (>40 dB drop), low-frequency drift rolloff (<0.5 Hz), and preservation of mu (8–12 Hz) and beta (13–30 Hz) bands. |
| **`03_erd_left_vs_right_psd.png`** | Overlaid PSD curves for Left vs Right hand trials on C3 and C4, evaluating contralateral Event-Related Desynchronization in sensorimotor cortex. |
| **`04_artifact_audit_distribution.png`** | Pie chart illustrating 47.7% clean vs 52.3% rejected trials, accompanied by KDE density distribution of log10 channel PTP amplitudes relative to the 200 uV cutoff. |
| **`05_channel_cross_correlation.png`** | Inter-channel correlation matrix heatmaps before vs after CAR. Proves CAR decouples shared environmental noise (raw correlation 0.93–0.98 drops to -0.27 between C3 and C4). |
| **`06_time_frequency_spectrogram_erd.png`** | Spectrograms of C3 (during Right hand imagery) and C4 (during Left hand imagery) across time (0.5–3.5s) and frequency (4–35 Hz). |

---

### 1.5 Modeling Package (`modeling/`)
| File | Role | Implementation Mechanics |
| :--- | :--- | :--- |
| **`modeling/__init__.py`** | Package initialization. | Exposes submodules for clean execution. |
| **`modeling/data_utils.py`** | Dataset loader and environment management. | Loads `X_clean.npy`, `X_clean_mubeta.npy`, and `y_clean.npy`, validates shapes, checks binary labels, and ensures output directories exist. |
| **`modeling/evaluation.py`** | Evaluation metrics and leakage prevention. | Sets up 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Computes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, out-of-fold (OOF) predictions, and generates confusion matrices and ROC curves. |
| **`modeling/train_classical.py`** | Classical baseline pipeline (CSP + SVM). | Fits `mne.decoding.CSP(n_components=4, log=True)` strictly on training folds, extracts log-variance features, and trains `sklearn.svm.SVC(kernel='linear', probability=True)`. |
| **`modeling/train_deep_learning.py`** | Deep learning training engine. | Implements PyTorch training loop with AdamW optimizer, Cross-Entropy loss, learning rate scheduling (`ReduceLROnPlateau`), early stopping (patience=10), and gradient clipping across 5-fold cross-validation. |
| **`modeling/audit_modeling.py`** | Rigorous data integrity and signal diagnostics. | Runs finite-value checks, duplicate checks, label alignment provenance verification, Welch spectral power t-tests between classes on C3/C4, and a 20-iteration label permutation test. |
| **`modeling/models/csp_svm.py`** | Classical model factory. | Constructs scikit-learn pipeline wrapping MNE CSP and SVM classifier. |
| **`modeling/models/eegnet.py`** | Compact convolutional neural network for EEG. | PyTorch implementation of EEGNet-8,2: 2D temporal convolution (`1 x 64`), depthwise spatial convolution across channels (`4 x 1`, depth multiplier=2), separable convolution (`1 x 16`), batch normalization, ELU activations, average pooling, and dropout. |
| **`modeling/models/cnn.py`** | Standard 2D convolutional baseline. | Multi-stage temporal and spatial convolutional network with feature flattening and fully-connected classification head. |
| **`modeling/models/cnn_lstm.py`** | Hybrid spatio-temporal recurrent network. | Cascades convolutional feature extraction stages into a recurrent LSTM layer (hidden size=32) to capture sequential temporal dynamics, followed by dense output. |
| **`modeling/models/transformer.py`** | Patch-based self-attention architecture. | Slices EEG epochs into temporal patches, projects to embedding dimension (64), adds learnable 1D positional embeddings, and passes through a 2-layer, 4-head Transformer Encoder with global average pooling. |
| **`modeling/MODEL_REPORT.md`** | Modeling technical documentation. | Documents input tensors, model definitions, leakage prevention protocols, and execution commands. |
| **`modeling/results/ROOT_CAUSE_ANALYSIS.md`** | Empirical root cause analysis report. | In-depth scientific investigation detailing why models achieved chance-level accuracy (~50-51%), supported by statistical power tests and permutation checks. |

---

### 1.6 Web Deployment Package (`Deployment/`)
| File | Role | Implementation Mechanics |
| :--- | :--- | :--- |
| **`Deployment/app.py`** | Flask web server backend. | Defines `/` (HTML UI), `/predict` (multipart POST endpoint ingesting user CSV, running `transform_single_trial()`, evaluating artifacts, and executing PyTorch EEGNet inference), and `/example` (GET endpoint running a pre-processed clean trial). |
| **`Deployment/templates/index.html`** | Front-end user interface markup. | Single-page UI with drag-and-drop file upload zone, dynamic status cards (idle, analyzing, success, rejected), prediction pill (`Left` / `Right`), and confidence progress meter. |
| **`Deployment/static/script.js`** | Client-side application logic. | Handles drag-and-drop file selection, AJAX `fetch` calls, loading spinners, dynamic DOM updates, and error state transitions. |
| **`Deployment/README.md`** | Deployment documentation. | Details rationale for Flask over Streamlit (custom UI control, lightweight API endpoints), endpoint schemas, setup commands, and response payloads. |

---

## 2. Section 1: Pre-processing

### 2.1 Overview & Domain Context
Raw Electroencephalography (EEG) is among the most notoriously noisy physiological signals recorded in biomedical engineering. Scalp electrodes capture volume-conducted postsynaptic potentials attenuated by the meninges, cerebrospinal fluid (CSF), skull bone, and scalp tissue, resulting in minute microvolt-level signals ($10–100\ \mu\text{V}$) contaminated by:
1. **Electrochemical Half-Cell Potentials & Electrode Drift**: Giant low-frequency DC biases ($>300,000$ ADC counts, equivalent to several millivolts) caused by skin-electrode impedance fluctuations and sweat.
2. **Powerline Interference**: 50 Hz electromagnetic radiation from the AC electrical grid coupling capacitively into electrode leads.
3. **Physiological Artifacts**: Ocular blinks (EOG, $100–500\ \mu\text{V}$), facial and neck muscle activation (EMG, high frequency $20–200\text{ Hz}$ bursts), and head motion/electrode displacement pops.

The mission of the preprocessing pipeline is to systematically isolate the **Sensorimotor Rhythms (SMR)** — specifically the **Mu band (8–12 Hz)** and **Beta band (13–30 Hz)** over the primary motor cortex (electrodes C3 and C4) — while discarding non-physiological noise.

---

### 2.2 Step-by-Step Design Choices, Rationale & Implementation

#### Step 1: Data Ingestion, Hardware Scaling & Uniform Time-Grid Resampling
* **Design Choice**: Convert raw 24-bit ADC counts to microvolts ($\mu\text{V}$) using $0.02235174\ \mu\text{V/count}$ and interpolate onto an exact uniform 250.0 Hz time grid.
* **Why Taken**:
  - The raw CSV files contained integer ADC counts around $300,000$. Without hardware calibration, artifact rejection thresholds (e.g. 200 uV) would incorrectly flag 99.5% of clean trials as corrupted. The hardware amplifier used is the Texas Instruments **ADS1299** (standard in OpenBCI Cyton 24-bit boards with Gain = 24 and $V_{\text{ref}} = 4.5\text{ V}$).
  - Hardware streaming over Bluetooth/WiFi introduces inter-sample jitter ($\Delta t$ varies between 0.000005s and 0.040s). Digital filtering (Butterworth, Notch) and Fourier/Wavelet analysis mathematically assume a strictly uniform sampling rate $F_s$.
* **How Implemented**:
  - In `preprocessing/loader.py`, signals are multiplied by $\frac{4.5}{24 \times (2^{23}-1)} \times 10^6 = 0.02235174\ \mu\text{V/count}$.
  - Timestamps are converted to relative elapsed time $t_{\text{rel}} = t - t_0$.
  - 1D linear interpolation (`scipy.interpolate.interp1d(..., fill_value='extrapolate')`) maps each channel onto an exact uniform grid: $t_k = \frac{k}{250.0}\text{ s}$ for $k \in [0, 2499]$, producing 2,500 points (9.996s duration).
* **Environment Relation**: Guarantees physical fidelity and temporal synchronization across all 2,160 trials.

#### Step 2: DC Offset Suppression & 50 Hz Powerline Notch Filter
* **Design Choice**: Linear detrending followed by a zero-phase 50 Hz IIR Notch filter with quality factor $Q = 30.0$.
* **Why Taken**:
  - Scalp electrodes establish an electrochemical half-cell potential with conductive paste, injecting massive DC offsets (~6.7 mV) and linear thermal drift. If passed into an IIR filter without detrending, filter edge ringing and step-response transients contaminate the first 1–2 seconds of data.
  - The recording environment operates on a 50 Hz AC electrical grid. The capacitive coupling manifests as a prominent, sharp 50 Hz peak that swamps low-amplitude cortical rhythms.
* **How Implemented**:
  - In `preprocessing/filters.py`, `scipy.signal.detrend(data, axis=-1, type='linear')` removes DC bias and linear slope.
  - An IIR notch filter is designed using `scipy.signal.iirnotch(50.0, 30.0, 250.0)`. Zero-phase bidirectional filtering (`scipy.signal.filtfilt`) attenuates the 50 Hz component by >40 dB with negligible phase distortion.
* **Environment Relation**: Cleans power grid contamination while preserving physiological bands below 45 Hz.

#### Step 3: Zero-Phase Butterworth Bandpass Filtering (0.5–40 Hz & 8–30 Hz)
* **Design Choice**: 4th-order zero-phase Butterworth filter implemented in Second-Order Sections (SOS). Two versions exported: Broadband (0.5–40 Hz) and Mu/Beta Band (8–30 Hz).
* **Why Taken**:
  - Slow drifts (<0.5 Hz) from sweat and breathing must be removed. Frequencies above 40 Hz contain cranial EMG muscle noise.
  - Motor imagery neurophysiology relies on Event-Related Desynchronization (ERD) in Mu (8–12 Hz) and Beta (13–30 Hz). Providing a dedicated 8–30 Hz band enables optimal covariance estimation for CSP, while 0.5–40 Hz broadband allows deep neural networks (EEGNet) to learn custom temporal filterbanks.
  - **Zero-phase filtering** is mandatory: causal IIR filters induce non-linear phase delays, shifting peak latency and corrupting temporal cue alignment. Forward-backward filtering (`sosfiltfilt`) has zero phase distortion.
* **How Implemented**:
  - `scipy.signal.butter(4, [0.5/125.0, 40.0/125.0], btype='bandpass', output='sos')` creates stable second-order sections.
  - Applied via `scipy.signal.sosfiltfilt(sos, data, axis=-1)`.
* **Environment Relation**: Stabilizes numerical calculations and prevents phase shifting of motor cortex activation.

#### Step 4: Spatial Re-referencing (Common Average Reference - CAR)
* **Design Choice**: Subtract the instantaneous across-channel mean at every sample: $V_i^{\text{CAR}}(t) = V_i(t) - \frac{1}{M}\sum_{j=1}^M V_j(t)$.
* **Why Taken**:
  - Unipolar EEG recordings measure voltage relative to an earlobe or mastoid reference electrode. Environmental electromagnetic interference and muscle activity at the reference electrode contaminate all recording channels identically.
  - CAR assumes that across the scalp surface, inward and outward dipole currents sum approximately to zero, canceling distant global common-mode noise and sharpening local cortical contrast.
* **How Implemented**:
  - In `preprocessing/re_referencing.py`, `mean = np.mean(data, axis=-2, keepdims=True)` followed by `data - mean`.
* **Environment Relation**: Unmasks local cortical activity; dramatically reduced inter-channel correlation from 0.98 to -0.27 between C3 and C4.

#### Step 5: Multi-Criterion Artifact Rejection & Scientific Audit
* **Design Choice**: Discard corrupted trials using a three-tier rule:
  1. Peak-to-Peak (PTP) amplitude $\le 200\ \mu\text{V}$.
  2. Channel variance $\ge 0.5\ \mu\text{V}^2$ and $<15$ consecutive flatline samples.
  3. Robust cohort variance $|Z| \le 4.0$.
* **Why Taken**:
  - Normal cognitive EEG rarely exceeds $100\ \mu\text{V}$. Voltage deflections $>200\ \mu\text{V}$ represent eye blinks, clenching jaws, or cable movement.
  - Dead electrodes or disconnected channels output flatline zeros or rail-clipped constants.
  - High-energy transient bursts distort covariance matrices in CSP and dominate gradient updates in neural networks.
  - Documenting every rejection reason in a CSV audit trail adheres to scientific transparency.
* **How Implemented**:
  - In `preprocessing/artifacts.py`, `evaluate_trial_artifacts()` calculates channel-wise PTP, variance, and max consecutive identical diffs.
  - Cohort variance Z-scores are computed using median and Median Absolute Deviation (MAD): $Z = 0.6745 \times \frac{\text{Var} - \text{Median}}{\text{MAD}}$.
  - Generates `trial_audit_metadata.csv` and `rejection_summary.json`.
* **Environment Relation**: Prevents non-cerebral noise artifacts from poisoning the machine learning models.

#### Step 6: Pre-Stimulus Baseline Correction & Task Epoching
* **Design Choice**: Subtract mean of pre-stimulus window (0.0–0.5s), slice task window (0.5–3.5s post-cue).
* **Why Taken**:
  - In motor imagery protocols, trials begin with a fixation cross (0–0.5s), followed by a visual cue indicating Left or Right hand imagery. Mental simulation occurs continuously across the 0.5–3.5s interval.
  - Subtracting the pre-cue baseline centers the epoch at $0\ \mu\text{V}$, removing residual low-frequency shifts.
* **How Implemented**:
  - In `preprocessing/epoching.py`, `apply_baseline_correction()` masks $t \in [0.0, 0.5]$ and subtracts the mean.
  - `extract_epoch_window()` slices $t \in [0.5, 3.5]$, yielding exactly 751 time points per channel ($3.0\text{ s} \times 250\text{ Hz} + 1$).
* **Environment Relation**: Aligns signal extraction strictly with cognitive task execution.

#### Step 7: Statistical Winsorization & Per-Channel Normalization
* **Design Choice**: Soft-clip amplitude values beyond $\pm 4.5\sigma$ (Winsorization) and apply per-channel Z-score standardization ($\mu=0, \sigma=1$).
* **Why Taken**:
  - Retained clean trials may still contain isolated single-sample glitches. Soft-clipping at $4.5\sigma$ conditions the distribution without truncating physiological waves.
  - Deep neural networks train stably when input features have zero mean and unit variance, preventing gradient explosion.
* **How Implemented**:
  - In `preprocessing/normalizer.py`, values outside $[\mu - 4.5\sigma, \mu + 4.5\sigma]$ are clipped.
  - Signals are standardized via $\frac{x - \text{mean}(x)}{\text{std}(x) + 10^{-8}}$.
* **Environment Relation**: Prepares standardized tensors for gradient descent optimization.

---

### 2.3 Preprocessing Results & Summary

```
============================================================
              PREPROCESSING AUDIT METRICS
============================================================
Total Ingested Trials        : 2,160
Clean Accepted Trials        : 1,031 (47.73%)
Rejected Artifact Trials     : 1,129 (52.27%)
Output Tensor Dimensions     : (1031, 4, 751) [Trials x Channels x Samples]
Target Sampling Frequency    : 250.0 Hz
Epoch Duration               : 3.004 seconds (751 samples)
------------------------------------------------------------
Clean Class Distribution:
  - Class 0 (LEFT Hand)      : 533 trials (51.70%)
  - Class 1 (RIGHT Hand)     : 498 trials (48.30%)
  - Ratio                    : 1.07 : 1.00 (Balanced)
------------------------------------------------------------
Primary Rejection Causes:
  - Extreme PTP Amplitude    : 982 trials (>200 uV, blinks/EMG)
  - Statistical Outlier (|Z|): 814 trials (|Z| > 4.0)
  - Dead Channel / Flatline  : 147 trials (Var < 0.5 uV²)
============================================================
```

---

## 3. Section 2: Exploratory Data Analysis (EDA)

### 3.1 Must-Do Integrity Checks & Findings
1. **Class Balance**: In the raw dataset of 2,160 trials, class distribution was 1,082 Left vs 1,078 Right (50.09% vs 49.91%). After strict artifact rejection, the retained 1,031 clean trials maintained a virtually perfect balance of 533 Left (51.7%) vs 498 Right (48.3%). No artificial rebalancing or class-weighting was necessary.
2. **Sampling Rate Consistency**: Timestamp differences revealed minor jitter in raw hardware capture (mean $\Delta t = 0.003997$s, min $0.000005$s, max $0.040152$s). Resampling onto a standardized 250.0 Hz grid completely eliminated temporal jitter.
3. **Missing Values & NaN Checks**: All 2,160 files had zero missing cells, zero NaNs, and zero infinite values.
4. **Channel Names**: Confirmed consistent 4-electrode montage across all trials: `FZ` (frontal midline), `C3` (left sensorimotor cortex), `CZ` (central midline), and `C4` (right sensorimotor cortex).

---

### 3.2 Visualizations & Neurophysiological Interpretations

#### 1. Time-Domain Signal Dynamics (`01_raw_vs_filtered_time_domain.png`)
* **Visual Observation**: Raw continuous signals exhibited massive microvolt offsets (~7,200 uV) with slow drifts and persistent 50 Hz fuzz. Clean epoched signals (0.5–3.5s) are zero-centered, smooth, and oscillatory within standard scalp EEG bounds.
* **Neurophysiological Takeaway**: Proves that detrending and bandpass filtering successfully extracted physiological rhythmic activity without baseline distortion.

#### 2. Power Spectral Density & Filter Verification (`02_psd_filtering_verification.png`)
* **Visual Observation**: Raw Welch PSD showed a sharp powerline spike at 50 Hz and a steep low-frequency 1/f noise floor. Preprocessed PSD displays a flat, clean passband between 0.5 Hz and 40 Hz, an attenuation of >40 dB at 50 Hz, and clear preservation of the Mu (8–12 Hz) and Beta (13–30 Hz) power bands.
* **Neurophysiological Takeaway**: Confirms that powerline noise is eliminated without attenuating the critical sensorimotor rhythm frequency bands.

#### 3. Left vs Right Motor Imagery ERD Spectral Contrast (`03_erd_left_vs_right_psd.png`)
* **Neurophysiological Theory**:
  - Imagining **Right Hand** movement activates the contralateral left motor cortex, causing an Event-Related Desynchronization (power *decrease*) in Mu/Beta rhythms on electrode **C3**.
  - Imagining **Left Hand** movement activates the contralateral right motor cortex, causing an ERD (power *decrease*) in Mu/Beta rhythms on electrode **C4**.
* **Empirical Observation in This Dataset**:
  - Plotting average Welch PSD on C3 and C4 reveals **almost identical power curves** for Left and Right trials.
  - Quantitative analysis in `class_signal_summary.csv` shows that the difference in mean Mu power on C3 between Left (0.1544) and Right (0.1528) is only **0.0016** ($p = 0.588$, non-significant).
  - On C4, the Mu power difference between Left (0.1553) and Right (0.1545) is only **0.0007** ($p = 0.800$, non-significant).
* **Key Insight**: There is no statistically significant contralateral power separation between classes when averaged across the cohort. This directly explains why downstream classifiers struggled to exceed chance level.

#### 4. Spatial Contrast Enhancement via CAR (`05_channel_cross_correlation.png`)
* **Visual Observation**:
  - **Before CAR**: Cross-channel correlation matrix was nearly uniform at $0.93–0.98$ across all channel pairs (`FZ`, `C3`, `CZ`, `C4`), reflecting massive common-mode reference noise.
  - **After CAR**: Cross-channel correlation dropped dramatically: `C3` and `C4` correlation decreased to **-0.27**, and `FZ` vs `CZ` dropped to **-0.72**.
* **Neurophysiological Takeaway**: CAR successfully removed shared reference noise, transforming global potential shifts into localized cortical differential signals.

#### 5. Time-Frequency Spectrogram Analysis (`06_time_frequency_spectrogram_erd.png`)
* **Visual Observation**: Spectrograms across the 0.5–3.5s window show transient bursts of energy in the 10–25 Hz range, but without sustained, time-locked power divergence between Left and Right trials.
* **Neurophysiological Takeaway**: Motor imagery execution across trials was temporally diffuse, likely due to latency jitter in subjects initiating mental imagery after the cue.

---

## 4. Section 3: Models & Empirical Evaluation

### 4.1 Modeling Strategy & Leakage Prevention
To evaluate motor imagery classification, five distinct architectures representing both classical spatial filtering and modern deep learning were benchmarked under rigorous cross-validation:

1. **CSP + SVM (Classical Baseline)**: Common Spatial Patterns spatial filter followed by Support Vector Machine.
2. **EEGNet (Specialized EEG Deep Learning)**: Compact convolutional network designed specifically for brain signals.
3. **CNN (Deep 2D Convolutional Network)**: General temporal-spatial convolutional baseline.
4. **CNN-LSTM (Spatio-Temporal Hybrid)**: Convolutional feature extractor cascaded into a Long Short-Term Memory recurrent network.
5. **Transformer (Self-Attention Network)**: Temporal patch tokenization with multi-head self-attention.

#### Leakage Prevention Protocol
- **Strict 5-Fold Stratified Cross-Validation**: Data split using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Zero Preprocessing Leakage**: Preprocessing was fit independently, or applied strictly per trial.
- **Inner Validation for Deep Models**: Each 80% training fold was subdivided into 75% train and 25% validation for early stopping (patience = 10 epochs). The 20% test fold was evaluated strictly once as held-out data.
- **CSP Fit Isolation**: CSP spatial projection filters were fit exclusively on the training fold; test fold epochs were transformed using the frozen training projection matrix.

---

### 4.2 Comprehensive Model Results Comparison

Below are the final out-of-fold cross-validated performance metrics across all 1,031 clean trials:

| Model Architecture | Input Data Band | Parameters | Test Accuracy (Mean ± Std) | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSP + SVM** | `X_clean_mubeta` (8–30 Hz) | — | **50.15% ± 0.74%** | 49.65% | 0.4713 | 0.3612 | 0.3866 | 0.5136 | 1.35s |
| **EEGNet** | `X_clean` (0.5–40 Hz) | 1,426 | **51.79% ± 1.63%** | **51.34%** | **0.4996** | 0.3525 | 0.3447 | **0.5170** | 15.57s |
| **CNN** | `X_clean` (0.5–40 Hz) | 26,834 | 50.72% ± 2.48% | 50.40% | 0.4869 | **0.4096** | **0.4221** | 0.4917 | 18.56s |
| **CNN-LSTM** | `X_clean` (0.5–40 Hz) | 37,202 | 50.92% ± 1.29% | 49.88% | 0.5708 | 0.1789 | 0.2033 | 0.4887 | 23.92s |
| **Transformer** | `X_clean` (0.5–40 Hz) | 44,402 | 50.34% ± 2.38% | 49.40% | 0.4610 | 0.2207 | 0.2831 | 0.4908 | 15.45s |

---

### 4.3 Diagnostic Audits & Permutation Tests

To verify that the near-50% accuracy was not caused by a coding bug, label mismatch, or data leakage, three diagnostic audits were executed:

1. **Label Permutation Test (Chance Baseline)**:
   - Shuffling labels across 20 iterations yielded an average CSP test accuracy of **50.50% ± 1.10%**.
   - Real-label CSP achieved **50.15% ± 0.74%**, which is statistically indistinguishable from random chance ($p = 0.55$).
2. **Train vs Validation vs Test Accuracy Tracking**:
   - Deep models showed slight training set memorization (CNN reached 63.5% train accuracy), but validation and test accuracy never exceeded 51.8–53.1%, proving lack of generalizable signal features.
3. **Label Alignment Audit**:
   - Provenance tracking confirmed that trial metadata, CSV row indices, and `.npy` arrays were 100% aligned with zero index offsets.

---

### 4.4 In-Depth Root Cause Analysis: Why Are the Results Not Good?

In scientific machine learning, understanding and documenting **why** a pipeline yields chance-level results is as critical as the pipeline itself. The ~50% accuracy is driven by fundamental neurophysiological, hardware, and algorithmic constraints inherent to this dataset:

#### 1. Extreme Spatial Sparsity (Only 4 Electrodes: FZ, C3, CZ, C4)
- Classical Motor Imagery BCI systems (e.g. BCI Competition IV Dataset 2a) rely on **22 to 64 electrodes**.
- Common Spatial Patterns (CSP) calculates spatial projection vectors that maximize variance for one class while minimizing it for the other. With only 4 channels, the spatial covariance matrix is $4 \times 4$, providing only **4 degrees of freedom**. There are simply not enough spatial dimensions for CSP to construct a spatial filter that cancels volume conduction from neighboring brain regions.
- Effective sensorimotor spatial isolation requires local **Laplacian rings** (e.g., surrounding C3 with FC3, CP3, C1, and C5). With only midline electrodes (FZ, CZ) available, spatial sharpening is severely constrained.

#### 2. Common Average Reference (CAR) Distortion with 4 Channels
- CAR subtracts the mean across all channels: $V_{\text{C3}}^{\text{CAR}} = V_{\text{C3}} - \frac{FZ + C3 + CZ + C4}{4} = \frac{3}{4}C3 - \frac{1}{4}C4 - \frac{1}{4}FZ - \frac{1}{4}CZ$.
- Because C3 and C4 are the primary contralateral sensorimotor electrodes, subtracting the average of only 4 channels forces one-quarter of C4's signal directly into C3, and vice versa! This inadvertently **attenuates the lateralized difference** between Left and Right hand imagery.

#### 3. Loss of Inter-Channel Power Discrepancies via Per-Trial Z-Scoring
- In `preprocessing/normalizer.py`, each channel of each trial was normalized to zero mean and **unit variance** ($\sigma = 1.0$).
- As documented in `diagnostics/class_signal_summary.csv`, this forced the variance and RMS of all channels on all trials to be exactly **1.000**.
- CSP is fundamentally a **variance-based algorithm**; it separates classes based on relative channel power changes. Forcing every channel to have variance = 1.0 flattened amplitude-dependent discriminant cues.

#### 4. Cross-Subject Pooling Without Subject Identifiers
- The 2,160 trials in this dataset appear to be pooled across multiple individuals without subject identifier tags.
- In EEG neuroscience, sensorimotor rhythms exhibit massive **inter-subject variability**:
  - The peak Mu frequency (Individual Alpha Frequency - IAF) varies between 8.5 Hz and 12.5 Hz across different people.
  - The physical location of the motor hand knob varies anatomically.
  - Training a single global model across pooled subjects without subject-specific calibration or domain adaptation washes out subtle individual ERD responses into the group noise floor.

#### 5. BCI Illiteracy Phenomenon
- In clinical EEG literature, **15% to 30% of human subjects are "BCI illiterate"** — meaning they do not produce measurable, detectable Mu/Beta desynchronization during motor imagery without extensive neurofeedback training over multiple sessions.
- If raw subjects recorded in this dataset were untrained or received no real-time sensory feedback, their mental imagery produced no consistent scalp potential modulations.

#### 6. High Artifact Corruption Rate (52.3% Rejection)
- 1,129 of 2,160 trials exceeded $200\ \mu\text{V}$ PTP or flatlined. This extraordinarily high noise contamination indicates poor electrode-skin contact impedance (>20 kΩ), excessive subject movement, jaw clenching, or eye blinks during acquisition.
- Even among retained clean trials, subtle sub-threshold muscle tension and ocular micro-saccades likely obscured the microvolt-level motor imagery rhythms.

---

## 5. Section 4: Deployment

### 5.1 Architecture & Implementation
The deployment system is implemented as a lightweight, production-grade **Flask web application** located in `Deployment/`:

```
User Web Browser (HTML/CSS/JS)
         │  
         │  HTTP POST /predict (Multipart CSV)
         ▼  HTTP GET  /example (Random Clean Trial)
┌────────────────────────────────────────────────────────┐
│ Flask Server (app.py)                                  │
│                                                        │
│ 1. Ingestion: load_single_trial(csv_file)              │
│    • Hardware calibration (counts -> µV)               │
│    • Uniform 250 Hz spline resampling (2500 samples)   │
│                                                        │
│ 2. Preprocessing: transform_single_trial()             │
│    • Detrending + 50 Hz Notch                          │
│    • 0.5–40 Hz Bandpass                                │
│    • CAR Re-referencing                                │
│    • Baseline Correction (0–0.5s)                      │
│    • Epoch Windowing (0.5–3.5s -> 751 samples)         │
│    • Winsorization (4.5σ) + Z-score Normalization      │
│                                                        │
│ 3. Artifact Safety Gate:                               │
│    • Check PTP <= 200 µV, Var >= 0.5 µV², |Z| <= 4.0   │
│    • If CORRUPTED: Halt & return rejection reason      │
│    • If CLEAN    : Proceed to neural inference         │
│                                                        │
│ 4. Neural Inference:                                   │
│    • Load weights: modeling/results/models/eegnet_best │
│    • PyTorch forward pass -> Softmax probabilities     │
│    • Return JSON: {"prediction": "Left", "confidence"} │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
Dynamic Web UI (Prediction Banner, Confidence Meter, Audit Alert)
```

---

### 5.2 Key Deployment Design Choices

1. **Why Flask Over Streamlit**:
   - While Streamlit is convenient for quick demos, Flask provides complete control over the DOM, CSS, and asynchronous JavaScript `fetch()` requests.
   - Enables a clean REST API interface (`/predict`, `/example`) that can be integrated into clinical hospital software or robotics systems.
2. **Artifact Safety Gate in Production**:
   - If a user uploads a trial contaminated by a cough, blink, or disconnected lead, the system **refuses to output an unreliable prediction**.
   - Instead, it immediately returns: `{"is_valid": false, "rejection_reasons": "extreme_ptp_amplitude_C3_(485.2uV > 200.0uV)"}`. This fail-safe design is mandatory in biomedical assistive technology.
3. **Model Selection for Serving**:
   - Deploys **EEGNet** (`eegnet_best.pt`), which achieved the highest test accuracy (51.8%) and has an ultralight memory footprint (only 1,426 parameters, <20 KB weight file), executing inference in under 5 milliseconds on CPU.

---

## 6. Section 5: Guide to Improving Results Across Every Section

To transition this BCI system from a chance-level baseline (~51%) to a high-accuracy system (>75–85%), actionable improvements must be executed across every stage of the pipeline:

---

### 6.1 Pre-processing Improvements
1. **Replace Per-Trial Channel Z-Scoring with Global / Baseline Normalization**:
   - *Problem*: Current per-trial Z-scoring forces every channel on every trial to variance = 1.0, destroying the spatial variance contrast that CSP and covariance classifiers depend on.
   - *Action*: Normalize signals relative to the **pre-stimulus baseline interval (0–0.5s)**: $x_{\text{norm}}(t) = \frac{x(t) - \mu_{\text{base}}}{\sigma_{\text{base}}}$, or perform global normalization across the entire training partition.
2. **Implement Local Bipolar / Targeted Laplacian Instead of CAR**:
   - *Problem*: 4-channel CAR subtracts C3 and C4 into each other.
   - *Action*: Compute local differential bipolar derivations: $C3_{\text{diff}} = C3 - CZ$ and $C4_{\text{diff}} = C4 - CZ$. This enhances the hemispheric dipole contrast without cross-talk.
3. **Advanced Artifact Repair via ICA or Wavelet Denoising**:
   - *Problem*: Dropping 52.3% of trials discards valuable training data.
   - *Action*: Apply Independent Component Analysis (FastICA or Infomax) or Wavelet Thresholding to decompose epochs, subtract blink/muscle components, and reconstruct clean EEG, rescuing 80–90% of rejected trials.
4. **Subject-Specific Frequency Tuning (Individual Alpha Frequency - IAF)**:
   - *Problem*: Fixed 8–12 Hz and 13–30 Hz filters miss subjects whose peak Mu rhythm is at 7.5 Hz or 12.8 Hz.
   - *Action*: Compute resting eyes-closed PSD for each subject to determine their exact IAF, centering filter passbands dynamically: $\text{Mu} = [\text{IAF}-2, \text{IAF}+2]\text{ Hz}$.

---

### 6.2 Exploratory Data Analysis (EDA) Improvements
1. **Event-Related Spectral Perturbation (ERSP) Maps**:
   - Compute log power ratios relative to pre-stimulus baseline over time: $\text{ERSP}(f, t) = 10 \log_{10}\left(\frac{P(f, t)}{P_{\text{base}}(f)}\right)$. This displays exactly when and at what frequency ERD peaks occur.
2. **Topographic Scalp Potential Mapping (Topoplots)**:
   - Interpolate 2D spherical spline topoplots across the scalp surface to visually confirm bilateral motor cortex activation patterns.
3. **Subject Heterogeneity Clustering**:
   - Use t-SNE or UMAP on channel spectral energy distributions to discover whether trials cluster into distinct participant sub-groups.

---

### 6.3 Modeling Improvements
1. **Filter Bank Common Spatial Pattern (FBCSP)**:
   - *Why*: State-of-the-art classical method (winner of BCI Competitions).
   - *Action*: Decompose EEG into multiple narrow sub-bands (e.g., 4–8, 8–12, 12–16, 16–20, ..., 36–40 Hz). Extract CSP features from each sub-band independently, apply Mutual Information Feature Selection (MIFS), and classify with SVM or Random Forest.
2. **Riemannian Geometry on Covariance Manifolds**:
   - *Why*: Avoids empirical spatial filtering issues by mapping trial covariance matrices $P_i = \frac{1}{T} X_i X_i^T$ onto the Riemannian manifold of Symmetric Positive Definite (SPD) matrices.
   - *Action*: Use `pyriemann` to compute Riemannian Tangent Space features followed by Logistic Regression, or use Minimum Distance to Mean (MDM) with Riemannian geodesic distance.
3. **Subject Transfer & Domain Adaptation**:
   - *Why*: Neutralizes cross-subject distribution shift.
   - *Action*: Implement Correlation Alignment (CORAL) or adversarial domain discriminators (DANN) to align source-subject feature representations with target-subject distributions.
4. **EEG Data Augmentation**:
   - *Why*: Deep neural networks overfit on small sample sizes ($N=1,031$).
   - *Action*: Apply time-reversal, Gaussian jittering, continuous segment recombination, and Mixup in the time domain to expand the training dataset 5-fold.

---

### 6.4 Data Acquisition & Hardware Improvements
1. **Increase Electrode Density to Sensorimotor Cluster (16–32 Channels)**:
   - *Action*: Upgrade from 4 electrodes to a focused 10–20 motor layout: `FC3`, `FC1`, `FCz`, `FC2`, `FC4`, `C5`, `C3`, `C1`, `Cz`, `C2`, `C4`, `C6`, `CP3`, `CP1`, `CPz`, `CP2`, `CP4`. This enables high-resolution true surface Laplacians.
2. **Active Electrodes & Impedance Verification**:
   - *Action*: Use active shielded electrodes (e.g. wet Ag/AgCl with conductive gel) and verify skin-contact impedance $<5\text{ k}\Omega$ before every recording session.
3. **Real-Time Neurofeedback Protocol**:
   - *Action*: Provide subjects with a visual cursor moving in real time with their detected Mu/Beta power. Subjects learn through operant conditioning to modulate sensorimotor rhythms within 2–3 sessions.
4. **Record Explicit Metadata**:
   - *Action*: Store Subject ID, Session Number, Cue Timestamp, and Trial Quality flags directly in file headers to enable subject-stratified cross-validation.

---

### 6.5 Deployment Improvements
1. **Sliding-Window Real-Time Streaming Pipeline**:
   - Transition from batch CSV uploads to real-time LSL (Lab Streaming Layer) buffer ingestion, updating predictions continuously every 250 ms using a 2.0-second sliding window.
2. **Confidence-Gated Feedback & Ambiguity Rejection**:
   - Reject predictions where softmax probability $<0.65$, holding the previous command rather than triggering false-positive robotic or assistive actuations.
3. **Model Quantization (ONNX Runtime)**:
   - Export PyTorch EEGNet to ONNX format and run with ONNX Runtime for sub-millisecond latency on edge microcontrollers (e.g. Raspberry Pi / NVIDIA Jetson).

---

## 7. Plain-English Defense & Presentation Guide (Simple Terms for Instructors)

This section provides intuitive, analogy-driven explanations of key engineering decisions in plain English. Use these explanations during your defense or presentation when your instructor asks *"Why did you do this?"* or *"What does this parameter actually mean?"*.

---

### 7.1 Step 1: Hardware ADC Scaling (0.02235 µV/count) & Uniform Resampling

#### 1. The Core Problem: Why Raw Numbers Break the Pipeline
* In raw CSV files, the signal readings look like `312450, 312520, ...` (integers around **$300,000$**).
* Project rule #4 requires rejecting trials with extreme amplitudes ($> 200\ \mu\text{V}$).
* If the code read $312,450$ and compared it directly to $200$, it would think every single signal is a massive artifact explosion, throwing **100% of the dataset into the trash**.
* **Reason**: $300,000$ is not microvolts; it is an internal digital hardware measurement called an **ADC count**.

#### 2. The Thermometer Analogy
* An electronic thermometer sensor might internally measure **$3,700$** raw sensor ticks.
* If it displayed $3,700$ on the screen, you would think you are boiling alive.
* But the thermometer simply multiplies by $0.01$ to convert internal ticks into Celsius: **$37.0^\circ\text{C}$**.
* That multiplier is the **Calibration / Scaling Factor**.

#### 3. What is an ADC and why 24-bit?
* **Brain Waves**: Tiny electrical oscillations on the scalp ($10–100\ \mu\text{V}$, where $1\ \mu\text{V} = 10^{-6}\text{ V}$).
* **ADC (Analog-to-Digital Converter)**: The electronic chip on the EEG headset that converts continuous analog voltages into digital integers.
* **24-bit Resolution**: Uses 24 binary bits ($2^{24}$ levels). Reserving 1 bit for the sign ($+$ or $-$) leaves **23 bits** for amplitude:
  $$2^{23} - 1 = 8,388,607\text{ discrete steps}$$
  The chip divides its measuring ruler into over **8 million tiny increments** to detect faint microvolt fluctuations.

#### 4. Hardware Parameters (ADS1299 & OpenBCI)
* The board is based on the Texas Instruments **ADS1299** biosensing chip (OpenBCI Cyton standard).
* Amplifier **Gain = 24** (magnifies the faint brain wave 24 times before digitizing).
* Reference Voltage **$V_{\text{ref}} = 4.5\text{ V}$**.

#### 5. Deriving the 0.02235174 µV/count Factor
The manufacturer's conversion formula from ADC counts to Volts is:
$$\text{Volts} = \frac{V_{\text{ref}}}{\text{Gain} \times (2^{23} - 1)} = \frac{4.5}{24 \times 8,388,607} = 0.00000002235174\text{ V}$$
Converting Volts to microvolts ($\times 10^6$):
$$\text{Scale Factor} = \mathbf{0.02235174\ \mu\text{V per count}}$$
Multiplying raw counts (~$300,000$) by this factor yields real physiological values (~$6,700\ \mu\text{V}$ total, with $20–50\ \mu\text{V}$ oscillatory brain dynamics after removing baseline).

#### 6. What is Timestamp Jitter & Uniform Resampling?
* Ideal sampling at $250\text{ Hz}$ means one sample arrives every **$0.004\text{ s}$** ($4\text{ ms}$).
* In wireless Bluetooth/WiFi streaming, packets arrive with small delays (e.g., $\Delta t = 0.00398\text{ s}$, then $0.00403\text{ s}$, sometimes $0.015\text{ s}$). This is **timestamp jitter**.
* **Why it breaks digital filters**: Butterworth filters, notch filters, and Fourier transforms (FFT) mathematically require a strictly uniform clock.
* **Fix**: Spline/linear interpolation resamples each trial onto an exact uniform 250.0 Hz time grid ($2,500$ points for $10.0\text{ s}$).

#### Instructor Quick Summary (Step 1)
> 1. Converted raw hardware counts (~$300,000$) into real microvolts ($\mu\text{V}$) via $0.02235\ \mu\text{V/count}$, preventing the $200\ \mu\text{V}$ artifact threshold from discarding the dataset.
> 2. The formula derives directly from the 24-bit ADS1299 amplifier specs ($4.5\text{V}$ reference, $24\times$ gain, $2^{23}-1$ steps).
> 3. Spline resampling eliminated wireless Bluetooth jitter so all digital filters run on an exact $250.0\text{ Hz}$ clock.

---

### 7.2 Step 2: DC Offset Suppression, Linear Detrending & 50 Hz Powerline Notch Filter

#### 1. The Chemistry: Electrochemical Half-Cell Potential (Accidental Battery)
* When a metal electrode ($\text{Ag/AgCl}$) touches conductive electrolyte gel and salty human skin, a chemical reaction occurs.
* This reaction forms a **miniature chemical battery** that generates a constant DC voltage of **~$6,700\ \mu\text{V}$ ($6.7\text{ mV}$)**.
* In contrast, real brain waves are only **$20–50\ \mu\text{V}$**.

> **The Flea on a Table Analogy:**  
> Trying to measure a $20\ \mu\text{V}$ brain wave on top of a $6,700\ \mu\text{V}$ DC offset is like trying to measure a **flea jumping $1\text{ millimeter}$ while standing on top of a $6\text{-meter tall table}$**. If your camera is zoomed in to see millimeters, you can't even see the flea because it is up at the ceiling!

#### 2. What is Linear Thermal Drift?
* As the subject sits in the chair, microscopic sweat, skin temperature changes, and gel drying cause the chemical battery voltage to slowly drift upward or downward over time (e.g., from $6,700\ \mu\text{V}$ to $6,900\ \mu\text{V}$).

#### 3. What is Linear Detrending and why do it *before* filtering?
* **Linear Detrending**: Fits a straight line ($y = mx + b$) to the drift and subtracts it, bringing the signal down from the "$6\text{-meter ceiling}$" so it oscillates cleanly around **$0\ \mu\text{V}$**.
* **Why do it before filtering? (Filter Edge Ringing)**:
  * Digital filters have mathematical inertia (like a pendulum).
  * If a filter starts at $t=0$ and is hit by a sudden jump from $0$ to $6,700\ \mu\text{V}$, it experiences an electronic shock.
  * It starts **oscillating wildly ("ringing like a struck bell")** before settling down. This is called a **step-response transient**, and it completely corrupts the first **1 to 2 seconds** of the trial.
  * Detrending first ensures the signal starts near zero, eliminating filter ringing.

#### 4. What is 50 Hz Powerline Hum?
* Wall wiring and electrical sockets oscillate at $50\text{ Hz}$ (AC power grid).
* Wires emit electromagnetic fields, and the human body and electrode cables act like **antennas** picking up this $50\text{ Hz}$ hum.
* It appears as a continuous, massive sine wave buzzing across every channel.

#### 5. What is an IIR Notch Filter and Quality Factor $Q = 30.0$?
* A **Notch Filter** is a specialized filter designed to carve out and delete **one single frequency** (50 Hz) while leaving everything else untouched.
* **Quality Factor ($Q$)**: Controls how sharp or blunt the cut is:
  $$\text{Bandwidth } (\Delta f) = \frac{f_0}{Q} = \frac{50\text{ Hz}}{30} \approx 1.67\text{ Hz}$$
  * Low $Q$ ($Q=2$): A blunt cut deleting $35–65\text{ Hz}$, destroying critical brain waves.
  * High $Q$ ($Q=30$): A razor-sharp sniper cut that deletes only **$49.2–50.8\text{ Hz}$** ($>40\text{ dB}$ drop), leaving sensorimotor rhythms below $45\text{ Hz}$ completely intact.

#### 6. What does "Zero-Phase" Mean?
* Normal electronic filters introduce a time delay (phase lag), shifting wave peaks by $50–100\text{ ms}$. In BCI, timing is everything.
* **Zero-Phase Filtering (`filtfilt`)**: Filters the signal **forward**, then flips it and filters it **in reverse**. The forward delay and reverse delay cancel out:
  $$\text{Total Delay} = (+\Delta t) + (-\Delta t) = \mathbf{0}$$
  The noise is removed, but the brain wave peaks stay at the **exact millisecond** they actually happened in the brain.

#### Instructor Quick Summary (Step 2)
> 1. **DC Offset**: Metal + gel + skin creates an accidental chemical battery (~$6.7\text{ mV}$). Detrending removes this huge bias so brain signals wiggle around $0\ \mu\text{V}$.
> 2. **Preventing Ringing**: Detrending first prevents digital filters from "ringing like a bell" and corrupting the first 1–2 seconds.
> 3. **50 Hz Notch ($Q=30$)**: A razor-sharp sniper filter that eliminates AC electrical wall hum (49.2–50.8 Hz) without harming brain rhythms.
> 4. **Zero-Phase**: Forward-backward filtering cancels out time delays, ensuring motor imagery event timing remains millisecond-accurate.

---

### 7.3 Complete Quick-Reference: All Filters & Algorithms by Name

When presenting to your instructor, use these precise scientific names. This demonstrates mastery of biomedical digital signal processing (DSP) and machine learning.

#### 1. Temporal & Frequency-Domain Filters
| Filter Name | Exact Scientific Name | Python Function / Package | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **IIR Notch Filter** | 2nd-Order Infinite Impulse Response Notch Filter | `scipy.signal.iirnotch` | Deletes the sharp $50\text{ Hz}$ AC powerline hum from electrical wiring ($Q=30.0$, bandwidth $\approx 1.67\text{ Hz}$). |
| **Butterworth Bandpass Filter** | 4th-Order Zero-Phase Butterworth Bandpass Filter (Second-Order Sections) | `scipy.signal.butter(..., output='sos')` | Suppresses slow DC drifts ($<0.5\text{ Hz}$) and muscle EMG noise ($>40\text{ Hz}$). Maximally flat passband with no ripple. |
| **Forward-Backward Filter (`filtfilt` / `sosfiltfilt`)** | Zero-Phase Digital Filtering (Gustafsson's Method) | `scipy.signal.sosfiltfilt` & `scipy.signal.filtfilt` | Passes signals forward then backward to cancel phase delay, ensuring zero time shift ($0\text{ ms}$ delay) for brain events. |
| **Linear Detrending Filter** | Ordinary Least Squares (OLS) Linear Trend Removal | `scipy.signal.detrend(type='linear')` | Removes linear electrochemical half-cell potential drift before filtering to prevent edge ringing. |

#### 2. Spatial Filters
| Filter Name | Exact Scientific Name | Mathematical Formula | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **CAR Filter** | Common Average Reference | $V_i^{\text{CAR}}(t) = V_i(t) - \frac{1}{M}\sum_{j=1}^M V_j(t)$ | Subtracts the instantaneous mean across all scalp electrodes to eliminate distant shared reference noise and unmask local cortical dynamics. |
| **Surface Laplacian Filter** | Hjorth Local Laplacian / Orthogonal Contrast Filter | $V_{\text{C3}}^{\text{Lap}} = V_{\text{C3}} - \frac{1}{K}\sum V_{\text{surrounding}}$ | Calculates the second spatial derivative of surface potential, acting as a spatial high-pass filter that eliminates volume conduction. |

#### 3. Preprocessing, Resampling & Statistical Algorithms
| Algorithm Name | Exact Scientific Name | Implementation | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Spline / Linear Interpolation** | 1D Continuous Piecewise Interpolation | `scipy.interpolate.interp1d` | Converts unevenly-spaced, jittered Bluetooth timestamps into an exact uniform $250.0\text{ Hz}$ temporal grid ($2,500$ samples). |
| **PTP Amplitude Thresholding** | Peak-to-Peak Extreme Value Detection | $\text{PTP} = \max(x) - \min(x)$ | Rejects trials where signal range exceeds $200\ \mu\text{V}$ (ocular blinks, head motion, electrode pops). |
| **Robust Z-Score Outlier Detection** | Median Absolute Deviation (MAD) Hampel Identifier | $Z = 0.6745 \times \frac{\text{Var} - \text{Median}(\text{Var})}{\text{MAD}}$ | Identifies non-Gaussian energy bursts and noisy trials without being distorted by extreme outliers. |
| **Winsorization** | Two-Tailed Statistical Winsorizing Soft-Clipper | Clips values to $[\mu - 4.5\sigma, \mu + 4.5\sigma]$ | Soft-clips isolated single-sample contact glitches in clean trials without truncating physiological waves. |
| **Z-Score Normalization** | Standard Gaussian Standardization | $z = \frac{x - \mu}{\sigma + \epsilon}$ | Standardizes each channel to zero mean and unit variance for neural network training stability. |

#### 4. Machine Learning & Deep Learning Algorithms
| Model / Algorithm Name | Exact Scientific Name | Architecture Details | Theoretical Function |
| :--- | :--- | :--- | :--- |
| **CSP** | Common Spatial Patterns | Generalized Eigenvalue Decomposition ($C_1 w = \lambda C_2 w$) | Maximizes signal variance for Class 1 (Left) while minimizing it for Class 2 (Right) in the $8–30\text{ Hz}$ band. |
| **SVM** | Support Vector Machine (Linear Kernel with Platt Scaling) | `sklearn.svm.SVC(kernel='linear', probability=True)` | Finds the optimal maximum-margin hyperplane separating CSP log-variance feature vectors. |
| **EEGNet** | Compact Convolutional Neural Network for EEG (EEGNet-8,2) | 2D Temporal Conv + Depthwise Spatial Conv + Separable Conv | Custom-tailored for brain signals; learns frequency filterbanks and virtual spatial electrode combinations. |
| **2D CNN** | Spatio-Temporal Convolutional Neural Network | Multi-stage temporal/spatial convolutions + dense head | General deep learning baseline learning temporal filters and spatial channel mixings. |
| **CNN-LSTM** | Spatio-Temporal Recurrent Hybrid Network | 2D CNN feature extractor $\to$ LSTM recurrent layer ($32$ units) | Captures spatial electrode correlations and sequential temporal dependencies over the $3.0\text{ s}$ window. |
| **Transformer** | Vision-Style Self-Attention Temporal Transformer | Patch Tokenization $\to$ Positional Encoding $\to$ 2-Layer 4-Head Encoder | Uses Multi-Head Self-Attention (MHSA) to model global long-range temporal dependencies across the epoch. |

#### 5. Validation & Scientific Evaluation Protocols
| Protocol Name | Scientific Definition | Implementation Details |
| :--- | :--- | :--- |
| **Stratified 5-Fold Cross-Validation** | Out-of-Fold (OOF) Stratified K-Fold | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ensuring 100% disjoint train/test splits with identical 50/50 class ratios in every fold. |
| **Label Permutation Test** | Non-Parametric Monte Carlo Permutation Test | Shuffles ground-truth labels across 20 iterations to compute the empirical null distribution, mathematically proving models operate at chance level ($50.50\% \approx 50.15\%$). |
| **Welch's Power Spectral Density (PSD)** | Welch's Averaged Modified Periodogram Method | `scipy.signal.welch` using overlapping Hanning windows to measure power distribution across frequency bands. |
