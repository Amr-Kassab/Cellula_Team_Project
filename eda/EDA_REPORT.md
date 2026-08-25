# EDA Report — BCI Motor Imagery Classification

**Team role:** EDA & Quality Assurance
**Input:** artifacts from the Data & Preprocessing stage (`../processed_data/`, `../PROCESSING_REPORT.md`)
**Script:** `eda_analysis.py` (run with `python eda/eda_analysis.py` from the repo root)

This EDA does not repeat the pipeline-verification plots already in `../figures/`
(raw-vs-filtered traces, PSD filter check, artifact pie chart, cross-correlation,
spectrogram). It focuses on the checks required by `document.pdf` §3: class
balance, sampling/quality audit, and Left-vs-Right group comparisons with
statistical tests.

## 1. Class balance
- Collected: **1082 Left / 1078 Right** (n=2160) — well balanced at the source.
- After the preprocessing pipeline's artifact rejection: **533 Left / 498 Right** (n=1031, 47.7% retained).
- Balance is preserved after cleaning (ratio ~1.07:1), so no class-imbalance correction is needed for modeling.
→ `figures/eda_01_class_balance.png`

## 2. Sampling consistency
- A raw sample trial has 2500 rows, 4 channels (FZ, C3, CZ, C4), **0 NaNs**.
- Raw inter-sample interval is highly non-uniform (CV ≈ 1.16, implied fs ≈ 250 Hz on average) — confirms the
  pipeline's resampling-to-250 Hz step was necessary before any frequency-domain analysis.
→ `figures/eda_02_sampling_jitter.png`

## 3. Rejection / quality audit
Of the 1129 rejected trials (52.3%), causes break down as (categories can overlap per trial):
- PTP amplitude outlier: 802
- Variance Z-score outlier: 630
- Flatline / dead channel: 325
→ `figures/eda_03_rejection_causes.png`, `figures/eda_04_variance_by_channel_class.png`

This rejection rate is high — worth flagging to the team/supervisor as a data-quality issue to
document (per the project brief's "scientific honesty" note), not just a pipeline detail.

## 4. Grand-average waveform (C3 / C4, Left vs Right)
Time-domain grand averages (± 1 SD) show heavily overlapping Left/Right traces on both channels —
no obvious visual separation in the broadband signal.
→ `figures/eda_05_grand_average_waveform.png`

## 5. Mu/Beta band-power — Left vs Right (Mann–Whitney U)
| Channel | Band | p-value | Left mean | Right mean |
|---|---|---|---|---|
| C3 | mu (8–12 Hz) | 0.51 | 0.0988 | 0.0987 |
| C3 | beta (13–30 Hz) | 0.14 | 0.3114 | 0.3182 |
| C4 | mu (8–12 Hz) | 0.72 | 0.0998 | 0.1005 |
| C4 | beta (13–30 Hz) | 0.24 | 0.3174 | 0.3205 |

None reach significance (all p > 0.05). → `figures/eda_06_bandpower_boxplots.png`, `bandpower_stats_summary.csv`

## 6. Contralateral vs ipsilateral ERD (Wilcoxon signed-rank)
Re-framed per-trial using the physiologically correct side (contralateral hemisphere = opposite
hand), expecting contra < ipsi power (event-related desynchronization):

| Band | p-value | Contra mean | Ipsi mean | Contra lower by |
|---|---|---|---|---|
| mu (8–12 Hz) | 0.65 | 0.0993 | 0.0996 | 0.32% |
| beta (13–30 Hz) | 0.59 | 0.3178 | 0.3158 | −0.63% |

No significant ERD effect detected either. → `erd_laterality_summary.csv`

## 7. PCA on band-power features
PC1 and PC2 explain ~47% and ~42% of variance respectively, but Left/Right classes show no visible
cluster separation in this 2D projection.
→ `figures/eda_07_pca_bandpower.png`

## Conclusion — is the data usable as-is?
Per §3.3 of the project brief: *"If no [PSD] difference appears, go back to processing or data
quality before modeling."* That is the situation here — none of the group comparisons (band power,
grand average, PCA, ERD laterality) show a significant or visually obvious Left/Right difference on
C3/C4 with the current preprocessing.

**Recommendation to the team, before moving to Modeling:**
1. Double-check epoch window alignment to the cue (0.5–3.5 s) and that trigger/label timing is correct.
2. Revisit the ~52% rejection rate — confirm thresholds in `preprocessing/artifacts.py` aren't overly
   aggressive (or too lenient) for this dataset's channel count (4 channels is sparse for CSP/ERD work).
3. Try channel-specific z-scoring or a narrower mu/beta filter before re-testing band power.
4. If differences remain absent after those checks, treat it as a real finding — CSP + SVM (Step 1 of
   the modeling roadmap) will show quickly whether a machine-learning model can pick up structure that
   these summary statistics can't, but expectations for accuracy should be set accordingly.
