from __future__ import annotations

import argparse
from pathlib import Path

from common.data_utils import INPUT_VOCAB
from common.logging_utils import create_logger
from common.models import LSTMRegressor
from common.run_helpers import build_regression_loaders, prepare_data
from common.train_eval import run_regression_training, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train LSTM regressor for addition NLP task.")
    parser.add_argument("--data-dir", type=str, default="data/addition")
    parser.add_argument("--out-dir", type=str, default="outputs/lstm_regression")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regen-data", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_df, val_df, test_df = prepare_data(args.data_dir, force_regen=args.regen_data, seed=args.seed)
    train_loader, val_loader, test_loader = build_regression_loaders(train_df, val_df, test_df, args.batch_size)

    model = LSTMRegressor(vocab_size=len(INPUT_VOCAB))
    logger = create_logger(Path(args.out_dir), name="lstm_regression")
    result, _ = run_regression_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger,
    )
    print(result)


if __name__ == "__main__":
    main()
