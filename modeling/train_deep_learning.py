"""Shared 5-fold training loop for compact neural EEG classifiers."""
from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from .data_utils import RESULTS_DIR, SEED, ensure_result_dirs, load_processed_data, set_global_seed
from .evaluation import compute_metrics, save_evaluation, save_training_curves
from .models import EEGNet, EEGCNN, CNNLSTM, EEGTransformer

MODEL_FACTORIES = {"eegnet": EEGNet, "cnn": EEGCNN, "cnn_lstm": CNNLSTM, "transformer": EEGTransformer}


def _loader(X, y, batch_size, shuffle=False):
    tensor_x = torch.from_numpy(X).float().unsqueeze(1)
    return DataLoader(TensorDataset(tensor_x, torch.from_numpy(y).long()), batch_size=batch_size, shuffle=shuffle)


def _evaluate(model, loader, device):
    model.eval(); total_loss = total_correct = total = 0; labels=[]; predictions=[]; probabilities=[]; loss_fn=torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); logits=model(xb); total_loss += loss_fn(logits,yb).item()*len(yb)
            p=torch.softmax(logits,1); labels.extend(yb.cpu().numpy()); predictions.extend(p.argmax(1).cpu().numpy()); probabilities.extend(p[:,1].cpu().numpy()); total += len(yb); total_correct += (p.argmax(1)==yb).sum().item()
    return total_loss/total, total_correct/total, np.array(labels), np.array(predictions), np.array(probabilities)


def run_deep_model(model_key: str, epochs: int = 60, batch_size: int = 32, patience: int = 10,
                   learning_rate: float = 1e-3, weight_decay: float = 1e-3) -> dict:
    if model_key not in MODEL_FACTORIES: raise ValueError(f"Unknown model: {model_key}")
    set_global_seed(); ensure_result_dirs(); X, y = load_processed_data("clean"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); oof_pred=np.empty(len(y),int); oof_prob=np.empty(len(y),float); oof_fold=np.empty(len(y),int); fold_metrics=[]; histories=[]; best_overall=None; started=time.perf_counter()
    for fold, (train_pool, test_idx) in enumerate(splitter.split(X,y), 1):
        train_idx, val_idx=train_test_split(train_pool, test_size=.15, stratify=y[train_pool], random_state=SEED+fold)
        # Model-level channel scaling is learned exclusively from this outer-training subset.
        # It neither alters nor writes the preprocessing arrays.
        channel_mean=X[train_idx].mean(axis=(0,2), keepdims=True)
        channel_std=X[train_idx].std(axis=(0,2), keepdims=True).clip(min=1e-6)
        X_train=(X[train_idx]-channel_mean)/channel_std
        X_val=(X[val_idx]-channel_mean)/channel_std
        X_test=(X[test_idx]-channel_mean)/channel_std
        model=MODEL_FACTORIES[model_key]().to(device); optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay); loss_fn=torch.nn.CrossEntropyLoss()
        train_loader=_loader(X_train,y[train_idx],batch_size,True); val_loader=_loader(X_val,y[val_idx],batch_size); test_loader=_loader(X_test,y[test_idx],batch_size)
        history={"train_loss":[],"val_loss":[],"train_accuracy":[],"val_accuracy":[]}; best_loss=float("inf"); best_state=None; stale=0; best_epoch=0
        for epoch in range(1, epochs+1):
            model.train(); loss_total=correct=seen=0
            for xb,yb in train_loader:
                xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(); logits=model(xb); loss=loss_fn(logits,yb); loss.backward(); optimizer.step()
                loss_total += loss.item()*len(yb); correct += (logits.argmax(1)==yb).sum().item(); seen += len(yb)
            val_loss,val_acc,*_= _evaluate(model,val_loader,device)
            history["train_loss"].append(loss_total/seen); history["train_accuracy"].append(correct/seen); history["val_loss"].append(val_loss); history["val_accuracy"].append(val_acc)
            if val_loss < best_loss - 1e-5:
                best_loss=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; stale=0
            else:
                stale += 1
                if stale >= patience: break
        model.load_state_dict(best_state); train_loss,train_acc,*_= _evaluate(model,_loader(X_train,y[train_idx],batch_size),device); val_loss,val_acc,*_= _evaluate(model,val_loader,device); _,_,true,pred,prob=_evaluate(model,test_loader,device); oof_pred[test_idx]=pred; oof_prob[test_idx]=prob; oof_fold[test_idx]=fold
        fold_metrics.append({"fold":fold, **compute_metrics(true,pred,prob), "best_epoch":best_epoch, "best_validation_loss":best_loss,
                             "train_loss_at_best":train_loss,"train_accuracy_at_best":train_acc,"validation_loss_at_best":val_loss,"validation_accuracy_at_best":val_acc,
                             "train_count":len(train_idx),"validation_count":len(val_idx),"test_count":len(test_idx,
                             )}); histories.append(history)
        if best_overall is None or best_loss < best_overall[0]: best_overall=(best_loss, fold, best_epoch, best_state)
    seconds=time.perf_counter()-started; parameter_count=sum(p.numel() for p in MODEL_FACTORIES[model_key]().parameters())
    _, fold, epoch, state=best_overall
    torch.save({"model_name":model_key, "state_dict":state, "input_shape":[1,4,751], "num_classes":2, "seed":SEED, "best_fold":fold, "best_epoch":epoch, "validation_loss":best_overall[0], "device_used":str(device), "fold_local_channel_normalization":True, "learning_rate":learning_rate, "weight_decay":weight_decay}, RESULTS_DIR/"models"/f"{model_key}_best.pt")
    display_name={"eegnet":"EEGNet", "cnn":"CNN", "cnn_lstm":"CNN-LSTM", "transformer":"Transformer"}[model_key]
    save_training_curves(display_name,histories)
    import pandas as pd
    pd.DataFrame(fold_metrics).to_csv(RESULTS_DIR / "metrics" / f"{model_key}_training_diagnostics.csv", index=False)
    return save_evaluation(display_name,y,oof_pred,oof_prob,oof_fold,fold_metrics,parameter_count,seconds)
