"""
Flask app for BCI Motor Imagery Classification deployment.
Loads the shared preprocessing pipeline + trained EEGNet model once at
startup, then serves two endpoints: /predict (upload a trial) and
/example (try a pre-processed clean trial without uploading anything).

Note: per modeling/results/ROOT_CAUSE_ANALYSIS.md, the models (including
EEGNet) were rigorously shown to not be learning real LEFT/RIGHT signal
from this dataset (real-label accuracy ~= shuffled-label accuracy in
permutation testing). Predictions and confidence scores below are real
model outputs, but should be presented honestly as "the best available
model" rather than as a working classifier.
"""

import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template

# --- make the parent repo folder importable (config.py, preprocessing/, modeling/) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.pipeline import EEGPreprocessingPipeline
from preprocessing.loader import load_single_trial
from modeling.models.eegnet import EEGNet

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load everything ONCE at startup (not per-request — these are expensive)
# ---------------------------------------------------------------------------
PIPELINE_PATH = os.path.join('..', 'processed_data', 'eeg_preprocessing_pipeline.joblib')
pipeline = EEGPreprocessingPipeline.load(PIPELINE_PATH)

# Load EEGNet: rebuild the architecture, then load the trained weights into it.
MODEL_PATH = os.path.join('..', 'modeling', 'results', 'models', 'eegnet_best.pt')
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = EEGNet(channels=4, classes=2)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # inference mode — disables dropout etc.

# EEGNet was trained on the broadband array (X_clean.npy), NOT the mu/beta
# filtered one — the pre-loaded example trial must match that.
X_CLEAN = np.load(os.path.join('..', 'processed_data', 'X_clean.npy'))
Y_CLEAN = np.load(os.path.join('..', 'processed_data', 'y_clean.npy'))
LABEL_MAP = {0: 'Left', 1: 'Right'}
N_CLEAN_TRIALS = len(X_CLEAN)


def predict_label(clean_epoch: np.ndarray):
    """
    Run a (4, 751) clean epoch through EEGNet and return (label, confidence).
    EEGNet expects input shape (batch, 1, channels, samples), so a single
    epoch needs two dimensions added: batch and the "1 input channel" dim
    Conv2d expects.
    """
    x = torch.tensor(clean_epoch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 751)
    with torch.no_grad():  # no gradients needed for inference
        logits = model(x)                     # (1, 2) raw scores
        probs = F.softmax(logits, dim=1)[0]    # convert to probabilities
    pred = int(torch.argmax(probs).item())
    confidence = float(probs[pred].item())
    return LABEL_MAP[pred], confidence


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Accept an uploaded raw trial CSV, preprocess it, and return a prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Could not read CSV: {e}'}), 400

    try:
        # Step 1: calibrate + resample raw CSV -> (4, 2500) array in uV
        signal, time_axis, load_meta = load_single_trial(df, config=pipeline.config)
        # Step 2: filter -> reref -> artifact check -> baseline -> epoch -> normalize
        clean_epoch, epoch_time, report = pipeline.transform_single_trial(signal, time_axis)
    except Exception as e:
        return jsonify({'error': f'Preprocessing failed: {e}'}), 400

    if not report['is_valid']:
        return jsonify({
            'is_valid': False,
            'rejection_reasons': report['rejection_reasons']
        })

    label, confidence = predict_label(clean_epoch)
    return jsonify({
        'is_valid': True,
        'prediction': label,
        'confidence': confidence
    })


@app.route('/example', methods=['GET'])
def example():
    """Run a random pre-processed clean trial through the model, no upload needed."""
    idx = random.randrange(N_CLEAN_TRIALS)  # different trial each call, for demo variety
    clean_epoch = X_CLEAN[idx]
    true_label = LABEL_MAP[int(Y_CLEAN[idx])]

    label, confidence = predict_label(clean_epoch)
    return jsonify({
        'is_valid': True,
        'prediction': label,
        'confidence': confidence,
        'true_label': true_label  # ground truth, useful for demo purposes
    })


if __name__ == '__main__':
    app.run(debug=True)