from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data_utils import OUTPUT_VOCAB, decode_seq
from .plot_utils import plot_roc_curve, plot_training_curves


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        out = []
        for item in batch:
            if torch.is_tensor(item):
                out.append(item.to(device))
            else:
                out.append(item)
        return out
    return batch.to(device)


def _compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    exact_match = (y_true == y_pred).all(axis=1).mean()
    digit_acc = (y_true == y_pred).mean()
    return {
        "exact_match": float(exact_match),
        "digit_accuracy": float(digit_acc),
    }


def _compute_sampled_multiclass_roc(
    y_true: np.ndarray,
    probs: np.ndarray,
    max_classes: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float] | Tuple[None, None, None]:
    uniq, counts = np.unique(y_true, return_counts=True)
    if len(uniq) < 2:
        return None, None, None

    top_idx = np.argsort(-counts)[:max_classes]
    selected = uniq[top_idx]

    mask = np.isin(y_true, selected)
    y_sel = y_true[mask]
    p_sel = probs[mask][:, selected]

    if len(np.unique(y_sel)) < 2:
        return None, None, None

    y_bin = label_binarize(y_sel, classes=selected)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
        p_sel = np.hstack([1 - p_sel, p_sel])

    try:
        auc = roc_auc_score(y_bin, p_sel, average="micro", multi_class="ovr")
    except ValueError:
        return None, None, None

    fpr, tpr, _ = roc_curve(y_bin.ravel(), p_sel.ravel())
    return fpr, tpr, float(auc)


def run_classifier_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    out_dir: str | Path,
    task_type: str,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
    logger=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}
    best_val = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        all_y_true, all_y_pred = [], []

        for batch in train_loader:
            batch = _to_device(batch, device)
            x, y = batch[0], batch[1]

            optimizer.zero_grad()
            logits = model(x)

            if task_type == "multiclass":
                loss = criterion(logits, y)
                pred = logits.argmax(dim=-1)
            elif task_type == "multilabel":
                loss = 0.0
                pred_parts = []
                for pos in range(logits.size(1)):
                    loss = loss + criterion(logits[:, pos, :], y[:, pos])
                    pred_parts.append(logits[:, pos, :].argmax(dim=-1, keepdim=True))
                pred = torch.cat(pred_parts, dim=1)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            all_y_true.append(y.detach().cpu().numpy())
            all_y_pred.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        train_metrics = (
            _compute_multiclass_metrics(y_true, y_pred)
            if task_type == "multiclass"
            else _compute_multilabel_metrics(y_true, y_pred)
        )

        val_loss, val_metrics, _, _ = evaluate_classifier(model, val_loader, task_type, device)

        main_key = "accuracy" if task_type == "multiclass" else "exact_match"
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(val_loss))
        history["train_metric"].append(float(train_metrics[main_key]))
        history["val_metric"].append(float(val_metrics[main_key]))

        if logger:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | train_%s=%.4f | val_%s=%.4f",
                epoch,
                epochs,
                history["train_loss"][-1],
                history["val_loss"][-1],
                main_key,
                history["train_metric"][-1],
                main_key,
                history["val_metric"][-1],
            )

        if val_metrics[main_key] > best_val:
            best_val = val_metrics[main_key]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics, test_probs, test_true = evaluate_classifier(
        model, test_loader, task_type, device, return_probs=True
    )

    roc_info = None
    if task_type == "multiclass" and test_probs is not None:
        fpr, tpr, auc = _compute_sampled_multiclass_roc(test_true, test_probs)
        if fpr is not None:
            roc_info = {"auc": auc}
            plot_roc_curve(fpr, tpr, auc, out_dir / "roc_curve.png", title="Sampled Multiclass ROC")

    plot_training_curves(history, out_dir / "training_curves.png", title=task_type)

    result = {
        "task_type": task_type,
        "test_loss": float(test_loss),
        **test_metrics,
    }
    if roc_info:
        result.update(roc_info)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(), out_dir / "best_model.pt")

    return result, history


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    task_type: str,
    device: str,
    return_probs: bool = False,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    all_y_true, all_y_pred = [], []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            x, y = batch[0], batch[1]
            logits = model(x)

            if task_type == "multiclass":
                loss = criterion(logits, y)
                pred = logits.argmax(dim=-1)
                probs = F.softmax(logits, dim=-1)
            else:
                loss = 0.0
                pred_parts = []
                for pos in range(logits.size(1)):
                    loss = loss + criterion(logits[:, pos, :], y[:, pos])
                    pred_parts.append(logits[:, pos, :].argmax(dim=-1, keepdim=True))
                pred = torch.cat(pred_parts, dim=1)
                probs = None

            losses.append(float(loss.item()))
            all_y_true.append(y.detach().cpu().numpy())
            all_y_pred.append(pred.detach().cpu().numpy())
            if probs is not None and return_probs:
                all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    metrics = (
        _compute_multiclass_metrics(y_true, y_pred)
        if task_type == "multiclass"
        else _compute_multilabel_metrics(y_true, y_pred)
    )
    probs_out = np.concatenate(all_probs, axis=0) if all_probs else None
    return float(np.mean(losses)), metrics, probs_out, y_true


def run_seq2seq_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    out_dir: str | Path,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
    logger=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}
    best_val = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_preds, train_tgts = [], []

        for batch in train_loader:
            src, tgt_in, tgt_out, raw_targets = _to_device(batch, device)
            optimizer.zero_grad()

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            pred_ids = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            train_preds.extend([decode_seq(ids) for ids in pred_ids])
            train_tgts.extend([str(x).lstrip("0") or "0" for x in raw_targets])

        train_exact = np.mean([p == t for p, t in zip(train_preds, train_tgts)])
        val_loss, val_metrics = evaluate_seq2seq(model, val_loader, device)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(val_loss))
        history["train_metric"].append(float(train_exact))
        history["val_metric"].append(float(val_metrics["exact_match"]))

        if logger:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | train_exact=%.4f | val_exact=%.4f",
                epoch,
                epochs,
                history["train_loss"][-1],
                history["val_loss"][-1],
                history["train_metric"][-1],
                history["val_metric"][-1],
            )

        if val_metrics["exact_match"] > best_val:
            best_val = val_metrics["exact_match"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate_seq2seq(model, test_loader, device)

    plot_training_curves(history, out_dir / "training_curves.png", title="seq2seq")
    result = {"task_type": "seq2seq", "test_loss": float(test_loss), **test_metrics}

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    return result, history


def evaluate_seq2seq(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []
    exact_flags = []
    char_acc_vals = []

    sos = OUTPUT_VOCAB["<sos>"]
    eos = OUTPUT_VOCAB["<eos>"]

    with torch.no_grad():
        for batch in loader:
            src, tgt_in, tgt_out, raw_targets = _to_device(batch, device)
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            losses.append(float(loss.item()))

            pred_ids = model.greedy_decode(src, sos_id=sos, eos_id=eos, max_len=tgt_out.size(1)).cpu().numpy().tolist()
            pred_strs = [decode_seq(ids) for ids in pred_ids]
            tgt_strs = [str(x).lstrip("0") or "0" for x in raw_targets]

            for p, t in zip(pred_strs, tgt_strs):
                exact_flags.append(1.0 if p == t else 0.0)
                p_digits = p.zfill(4)[:4]
                t_digits = t.zfill(4)[:4]
                char_acc_vals.append(np.mean([pc == tc for pc, tc in zip(p_digits, t_digits)]))

    metrics = {
        "exact_match": float(np.mean(exact_flags) if exact_flags else 0.0),
        "char_accuracy": float(np.mean(char_acc_vals) if char_acc_vals else 0.0),
    }
    return float(np.mean(losses)), metrics
