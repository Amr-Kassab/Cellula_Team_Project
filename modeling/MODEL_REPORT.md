# Modeling and Evaluation Report

## Scope and inputs

This isolated package performs modeling on read-only preprocessing outputs. CSP + SVM reads `processed_data/X_clean_mubeta.npy`, because Mu/Beta rhythms are informative for motor imagery and CSP learns discriminative spatial variance patterns in that band. EEGNet, CNN, CNN-LSTM, and Transformer read `processed_data/X_clean.npy`. Labels come from `processed_data/y_clean.npy`: `0 = LEFT`, `1 = RIGHT`.

## Models

- **CSP + SVM** fits CSP (`n_components=4`, log-variance output) and a linear probabilistic SVM independently in each fold.
- **EEGNet** uses temporal convolution, depthwise spatial convolution, separable convolution, normalization, pooling, and dropout; this is a compact design tailored to EEG's temporal and channel structure.
- **CNN** learns temporal and spatial filters with a small convolutional feature extractor.
- **CNN-LSTM** converts CNN output from `[batch, features, 1, time]` to `[batch, time, features]` before the LSTM models its feature sequence.
- **Transformer** embeds short temporal patches, adds learnable positional embeddings, and applies a lightweight two-layer, four-head Transformer encoder.

## Evaluation and leakage prevention

All experiments use seed 42 and five-fold `StratifiedKFold(shuffle=True, random_state=42)`. The held-out fold is used only for testing. Neural-model training folds are split again into stratified training and validation portions; validation drives early stopping. CSP is fit only on each training fold. The reported final scores are calculated from combined out-of-fold predictions.

Metrics are accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and a LEFT/RIGHT confusion matrix. Per-fold means and standard deviations, OOF predictions, ROC figures, confusion matrices, checkpoints, and training curves are written beneath `modeling/results/`.

## Running experiments

From the repository root, install the modeling-only dependencies if required, then run one model or all models:

```bash
pip install -r modeling/requirements_modeling.txt
python run_modeling.py --model csp_svm
python run_modeling.py --model eegnet
python run_modeling.py --model all
```

Use `--epochs` to set the maximum epochs for deep models. Actual results are intentionally not claimed here before experiments run; after runs, consult `results/metrics/model_comparison.csv` and each model's JSON/CSV output.
