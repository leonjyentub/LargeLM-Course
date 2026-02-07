from __future__ import annotations

import argparse
from pathlib import Path

from common.data_utils import INPUT_VOCAB
from common.logging_utils import create_logger
from common.models import FFNNClassifier
from common.run_helpers import build_classification_loaders, prepare_data
from common.train_eval import run_classifier_training, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train Simple FFNN classifier for addition NLP task.")
    parser.add_argument("--data-dir", type=str, default="data/addition")
    parser.add_argument("--out-dir", type=str, default="outputs/ffnn_classifier")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regen-data", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_df, val_df, test_df = prepare_data(args.data_dir, force_regen=args.regen_data, seed=args.seed)
    train_loader, val_loader, test_loader = build_classification_loaders(train_df, val_df, test_df, args.batch_size)

    model = FFNNClassifier(vocab_size=len(INPUT_VOCAB))
    logger = create_logger(Path(args.out_dir), name="ffnn")
    result, _ = run_classifier_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        out_dir=args.out_dir,
        task_type="multiclass",
        epochs=args.epochs,
        lr=args.lr,
        logger=logger,
    )
    print(result)


if __name__ == "__main__":
    main()
