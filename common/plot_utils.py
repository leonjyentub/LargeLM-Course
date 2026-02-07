from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_training_curves(history: dict, out_path: str | Path, title: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(x=list(epochs), y=history["train_loss"], ax=axes[0], label="train_loss")
    sns.lineplot(x=list(epochs), y=history["val_loss"], ax=axes[0], label="val_loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")

    sns.lineplot(x=list(epochs), y=history["train_metric"], ax=axes[1], label="train_metric")
    sns.lineplot(x=list(epochs), y=history["val_metric"], ax=axes[1], label="val_metric")
    axes[1].set_title(f"{title} - Metric")
    axes[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(fpr, tpr, auc_value: float, out_path: str | Path, title: str = "ROC Curve"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC={auc_value:.4f}")
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
