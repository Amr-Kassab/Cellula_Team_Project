# Processed Data Integrity Audit

All arrays were loaded read-only from `processed_data`; no preprocessing was run.

| Check | Finding |
|---|---|
| X_clean shape | (1031, 4, 751) |
| X_clean_mubeta shape | (1031, 4, 751) |
| y_clean shape | (1031,) |
| trial counts match | True |
| valid labels | True |
| class counts (LEFT, RIGHT) | [533, 498] |
| X_clean NaN / Inf | 0 / 0 |
| Mu/Beta NaN / Inf | 0 / 0 |
| exact duplicate X_clean trials | 9 |
| exact duplicate Mu/Beta trials | 9 |
| X_clean channel variances | {'FZ': np.float32(1.0), 'C3': np.float32(1.0), 'CZ': np.float32(1.0), 'C4': np.float32(1.0)} |
| flat channels | [] |
| metadata accepted trials | 1031 |
| metadata accepted label counts | {0: 533, 1: 498} |
