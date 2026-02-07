from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

METHODS = [
    ("ffnn", "train_ffnn_classifier.py", "outputs/ffnn_classifier/metrics.json"),
    ("lstm_cls", "train_lstm_classifier.py", "outputs/lstm_classifier/metrics.json"),
    ("gru_cls", "train_gru_classifier.py", "outputs/gru_classifier/metrics.json"),
    ("lstm_multilabel", "train_lstm_multilabel.py", "outputs/lstm_multilabel/metrics.json"),
    ("self_attention", "train_self_attention_classifier.py", "outputs/self_attention_classifier/metrics.json"),
    ("lstm_seq2seq", "train_lstm_seq2seq.py", "outputs/lstm_seq2seq/metrics.json"),
    ("transformer_seq2seq", "train_transformer_seq2seq.py", "outputs/transformer_seq2seq/metrics.json"),
]


def run_methods(epochs: int, batch_size: int, lr: float, data_dir: str):
    for _, script, _ in METHODS:
        cmd = [
            sys.executable,
            script,
            "--data-dir",
            data_dir,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def collect_results():
    rows = []
    for method, _, metrics_path in METHODS:
        path = Path(metrics_path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"method": method}
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No metrics.json found. Please run training scripts first.")
    return rows


def write_csv(rows, out_file: Path):
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    candidates = ["accuracy", "exact_match", "digit_accuracy", "char_accuracy", "macro_f1", "auc"]
    methods = [r["method"] for r in rows]

    metric_data = {m: [] for m in candidates}
    for m in candidates:
        for r in rows:
            metric_data[m].append(r.get(m, np.nan))

    x = np.arange(len(methods))
    valid_metrics = [m for m in candidates if np.isfinite(np.array(metric_data[m], dtype=float)).any()]

    width = 0.8 / max(len(valid_metrics), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(valid_metrics):
        vals = np.array(metric_data[metric], dtype=float)
        ax.bar(x + i * width, vals, width=width, label=metric)

    ax.set_xticks(x + width * max(len(valid_metrics) - 1, 0) / 2)
    ax.set_xticklabels(methods, rotation=30)
    ax.set_ylim(0, 1.0)
    ax.set_title("Method Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "all_methods_comparison.png", dpi=150)
    plt.close(fig)


def print_rank(rows):
    score_keys = ["accuracy", "exact_match", "char_accuracy"]

    def key_fn(r):
        return tuple(float(r.get(k, -1.0)) for k in score_keys)

    ranked = sorted(rows, key=key_fn, reverse=True)
    for i, r in enumerate(ranked, start=1):
        summary = ", ".join(
            [f"{k}={float(r.get(k, float('nan'))):.4f}" for k in ["accuracy", "exact_match", "char_accuracy", "macro_f1", "auc"] if k in r]
        )
        print(f"{i:>2}. {r['method']}: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Compare all NLP addition methods.")
    parser.add_argument("--run", action="store_true", help="Run all training scripts before comparison.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/addition")
    parser.add_argument("--out-dir", type=str, default="outputs/comparison")
    args = parser.parse_args()

    if args.run:
        run_methods(args.epochs, args.batch_size, args.lr, args.data_dir)

    rows = collect_results()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_dir / "all_methods_metrics.csv")
    plot_comparison(rows, out_dir)
    print_rank(rows)


if __name__ == "__main__":
    main()
