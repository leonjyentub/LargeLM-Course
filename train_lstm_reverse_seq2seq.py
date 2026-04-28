from __future__ import annotations

import argparse
from pathlib import Path

from common.logging_utils import create_logger
from common.models import LSTMReverseSeq2Seq
from common.run_helpers import build_seq2seq_loaders, prepare_data, vocab_sizes
from common.train_eval import run_seq2seq_training, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Seq2Seq that emits answer digits right-to-left.")
    parser.add_argument("--data-dir", type=str, default="data/addition")
    parser.add_argument("--out-dir", type=str, default="outputs/lstm_reverse_seq2seq")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regen-data", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_df, val_df, test_df = prepare_data(args.data_dir, force_regen=args.regen_data, seed=args.seed)
    train_loader, val_loader, test_loader = build_seq2seq_loaders(
        train_df,
        val_df,
        test_df,
        args.batch_size,
        reverse=True,
    )

    src_vocab_size, tgt_vocab_size = vocab_sizes()
    model = LSTMReverseSeq2Seq(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    logger = create_logger(Path(args.out_dir), name="lstm_reverse_seq2seq")
    result, _ = run_seq2seq_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        reverse_output=True,
        logger=logger,
    )
    print(result)


if __name__ == "__main__":
    main()
