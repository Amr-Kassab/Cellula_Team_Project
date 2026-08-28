# BCI Motor Imagery — Deployment

Flask web app that serves the team's EEG motor imagery pipeline: upload a raw
trial CSV (or try a random pre-processed example) and get a LEFT/RIGHT
prediction from the trained EEGNet model.

## Why Flask

Chosen over Streamlit for full control over the HTML/CSS/JS — this app uses a
custom-designed UI (drag-and-drop upload, idle/loading/success/rejected
states, live confidence bar) rather than Streamlit's built-in widgets. Flask
also keeps the app as a small, clear set of routes that map directly onto
the two things a user can do: upload a trial or try an example.

## Model used

**EEGNet** (`modeling/models/eegnet.py`, weights in
`modeling/results/models/eegnet_best.pt`) — the best-performing model out of
the five the team compared (CSP+SVM, EEGNet, CNN, CNN-LSTM, Transformer).

## How it works

```
Raw trial CSV
   -> preprocessing.loader.load_single_trial()      (calibrate + resample)
   -> preprocessing.pipeline.transform_single_trial() (filter, artifact
      check, baseline, epoch, normalize)
   -> EEGNet (if the trial passes artifact validation)
   -> LEFT / RIGHT + confidence
```

If a trial fails artifact validation (extreme amplitude, dead channel, etc.)
the app shows the rejection reason instead of forcing a prediction.

## Folder structure

```
Deployment/
├── app.py              # Flask routes + model/pipeline loading
├── requirements.txt
├── templates/
│   └── index.html      # UI markup
└── static/
    └── script.js        # upload handling, fetch calls, state toggling
```

Relies on shared repo resources one level up: `preprocessing/`, `config.py`,
`processed_data/`, and `modeling/`.

## Setup

From inside `Deployment/`:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the UI |
| `/predict` | POST | Accepts an uploaded CSV trial (`file` form field), preprocesses it, returns a prediction or rejection reason |
| `/example` | GET | Runs a random pre-processed clean trial through the model, no upload needed |

### Response shapes

Success:
```json
{"is_valid": true, "prediction": "Left", "confidence": 0.53}
```

Rejected:
```json
{"is_valid": false, "rejection_reasons": "extreme_ptp_amplitude_C3_(...)"}
```
