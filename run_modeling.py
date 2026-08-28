"""Run EEG modeling only; this script never invokes preprocessing."""
from __future__ import annotations
import argparse
from modeling.data_utils import ensure_result_dirs
from modeling.train_classical import run_csp_svm
from modeling.train_deep_learning import run_deep_model

MODELS=("csp_svm","eegnet","cnn","cnn_lstm","transformer")
def main():
    parser=argparse.ArgumentParser(description="Cross-validated motor imagery EEG modeling")
    parser.add_argument("--model", choices=(*MODELS,"all"), required=True)
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs per deep-learning fold")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--audit", action="store_true", help="Run read-only data/model diagnostic audit before requested models")
    args=parser.parse_args(); ensure_result_dirs()
    if args.audit:
        from modeling.audit_modeling import run_audit
        run_audit()
    selected=MODELS if args.model=="all" else (args.model,)
    for model in selected:
        print(f"\nRunning {model}...")
        result=run_csp_svm() if model=="csp_svm" else run_deep_model(model, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        print(result["oof_metrics"])
if __name__ == "__main__": main()
