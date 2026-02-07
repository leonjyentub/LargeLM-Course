from __future__ import annotations

import argparse

from common.data_utils import generate_split_csv


def main():
    parser = argparse.ArgumentParser(description="Generate NLP addition train/val/test splits.")
    parser.add_argument("--data-dir", type=str, default="data/addition")
    parser.add_argument("--n-train", type=int, default=80000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = generate_split_csv(
        out_dir=args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(f"train: {paths.train_csv}")
    print(f"val: {paths.val_csv}")
    print(f"test: {paths.test_csv}")


if __name__ == "__main__":
    main()
