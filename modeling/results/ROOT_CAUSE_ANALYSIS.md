# Root Cause Analysis

## 1. Is the data valid?

The data-integrity audit passed shape, binary-label, finite-value, non-zero-variance, and exact-duplicate checks. Full details are in `data_audit_report.md`.

## 2. Is label alignment verified?

**VERIFIED (by preprocessing provenance and metadata order).** The same validity mask indexes signals and encoded labels, and all 1031 accepted metadata labels match `y_clean`. The NPY arrays do not retain trial IDs, so this cannot independently reconstruct correspondence from signal contents.

## 3. Is there measurable LEFT/RIGHT difference?

C3/C4 Mu/Beta descriptive statistics are in `diagnostics/class_signal_summary.csv`; the observed class differences are tiny and their uncorrected p-values are non-significant. These descriptive tests are not confirmatory because multiple channel/measure comparisons are made.

## 4. Are models learning?

Sanity baseline results are in `metrics/sanity_check_results.csv`; nested CSP test-fold scores are in `metrics/csp_svm_nested_fold_metrics.csv`. 

| Model | Train accuracy | Validation accuracy | Held-out test accuracy |
|---|---:|---:|---:|
| EEGNet | 0.542 | 0.531 | 0.518 |
| CNN | 0.635 | 0.524 | 0.507 |
| CNN-LSTM | 0.551 | 0.521 | 0.509 |
| Transformer | 0.565 | 0.482 | 0.503 |

## 5. Real vs shuffled labels

Nested real-label CSP mean outer accuracy was 0.502; shuffled-label mean was 0.505 (SD 0.011, 20 permutations). This comparison is a sanity check, not proof of a causal mechanism.

## 6. Most likely explanation for ~50% accuracy

1. Weak or inconsistent class-discriminative information in this particular 3-second, four-channel trial-level dataset is the leading explanation: C3/C4 Mu/Beta class means are almost identical and real-label CSP does not exceed shuffled-label CSP.
2. Strong per-trial normalization/winsorization and the high 52.3% rejection rate may reduce stable amplitude cues; this is a provenance observation, not a recommendation to change preprocessing.
3. A label-order mismatch is less likely: alignment is supported by code and all retained metadata labels.
4. An obvious CSP split/metric bug is less likely after the nested audit records disjoint folds and held-out predictions.

No result is hidden: the complete numeric artifacts are retained under `modeling/results`.
