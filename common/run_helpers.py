from __future__ import annotations

from pathlib import Path

from .data_utils import (
    AdditionClassificationDataset,
    AdditionMultiLabelDataset,
    AdditionRegressionDataset,
    AdditionSeq2SeqDataset,
    INPUT_VOCAB,
    OUTPUT_VOCAB,
    build_dataloader,
    generate_split_csv,
    load_splits,
)


def prepare_data(data_dir: str | Path, force_regen: bool = False, seed: int = 42):
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    if force_regen or not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        generate_split_csv(data_dir, seed=seed)

    return load_splits(data_dir)


def build_classification_loaders(train_df, val_df, test_df, batch_size=256):
    train_ds = AdditionClassificationDataset(train_df)
    val_ds = AdditionClassificationDataset(val_df)
    test_ds = AdditionClassificationDataset(test_df)
    return (
        build_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        build_dataloader(val_ds, batch_size=batch_size, shuffle=False),
        build_dataloader(test_ds, batch_size=batch_size, shuffle=False),
    )


def build_regression_loaders(train_df, val_df, test_df, batch_size=256):
    train_ds = AdditionRegressionDataset(train_df)
    val_ds = AdditionRegressionDataset(val_df)
    test_ds = AdditionRegressionDataset(test_df)
    return (
        build_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        build_dataloader(val_ds, batch_size=batch_size, shuffle=False),
        build_dataloader(test_ds, batch_size=batch_size, shuffle=False),
    )


def build_multilabel_loaders(train_df, val_df, test_df, batch_size=256):
    train_ds = AdditionMultiLabelDataset(train_df)
    val_ds = AdditionMultiLabelDataset(val_df)
    test_ds = AdditionMultiLabelDataset(test_df)
    return (
        build_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        build_dataloader(val_ds, batch_size=batch_size, shuffle=False),
        build_dataloader(test_ds, batch_size=batch_size, shuffle=False),
    )


def build_seq2seq_loaders(train_df, val_df, test_df, batch_size=256, reverse: bool = False):
    train_ds = AdditionSeq2SeqDataset(train_df, reverse=reverse)
    val_ds = AdditionSeq2SeqDataset(val_df, reverse=reverse)
    test_ds = AdditionSeq2SeqDataset(test_df, reverse=reverse)
    return (
        build_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        build_dataloader(val_ds, batch_size=batch_size, shuffle=False),
        build_dataloader(test_ds, batch_size=batch_size, shuffle=False),
    )


def vocab_sizes():
    return len(INPUT_VOCAB), len(OUTPUT_VOCAB)
