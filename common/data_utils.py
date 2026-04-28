from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # allow data generation without torch installed
    torch = None

    class Dataset:  # type: ignore
        pass

    DataLoader = object  # type: ignore

MAX_INPUT_LEN = 7   # "999+999"
MAX_OUTPUT_DIGITS = 4  # 1998
MAX_SEQ_LEN = 5    # <sos> + 4 digits OR 4 digits + <eos>

INPUT_CHARS = ["+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
OUTPUT_CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

INPUT_VOCAB = {"<pad>": 0, **{ch: idx + 1 for idx, ch in enumerate(INPUT_CHARS)}}
OUTPUT_VOCAB = {"<pad>": 0, "<sos>": 1, "<eos>": 2, **{ch: idx + 3 for idx, ch in enumerate(OUTPUT_CHARS)}}
OUTPUT_ID2TOK = {v: k for k, v in OUTPUT_VOCAB.items()}


@dataclass
class DatasetPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path


def _idx_to_pair(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = idx // 999 + 1
    b = idx % 999 + 1
    return a.astype(np.int32), b.astype(np.int32)


def _save_csv(path: Path, rows: Sequence[Tuple[int, int, str, int]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "expr", "sum"])
        writer.writerows(rows)


def generate_split_csv(
    out_dir: str | Path,
    n_train: int = 80000,
    n_val: int = 10000,
    n_test: int = 10000,
    seed: int = 42,
) -> DatasetPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = 999 * 999
    needed = n_train + n_val + n_test
    if needed > total_pairs:
        raise ValueError(f"Requested {needed} samples > all unique pairs {total_pairs}.")

    rng = np.random.default_rng(seed)
    all_indices = rng.choice(total_pairs, size=needed, replace=False)

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train : n_train + n_val]
    test_idx = all_indices[n_train + n_val :]

    def build_rows(idxs: np.ndarray) -> List[Tuple[int, int, str, int]]:
        a, b = _idx_to_pair(idxs)
        s = a + b
        expr = np.char.add(np.char.add(a.astype(str), "+"), b.astype(str))
        return [(int(x), int(y), str(e), int(z)) for x, y, e, z in zip(a, b, expr, s)]

    train_rows = build_rows(train_idx)
    val_rows = build_rows(val_idx)
    test_rows = build_rows(test_idx)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    _save_csv(train_path, train_rows)
    _save_csv(val_path, val_rows)
    _save_csv(test_path, test_rows)

    return DatasetPaths(train_csv=train_path, val_csv=val_path, test_csv=test_path)


def _load_split(path: Path) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row["expr"], int(row["sum"])))
    return samples


def load_splits(data_dir: str | Path) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    data_dir = Path(data_dir)
    return (
        _load_split(data_dir / "train.csv"),
        _load_split(data_dir / "val.csv"),
        _load_split(data_dir / "test.csv"),
    )


def encode_input(expr: str, max_len: int = MAX_INPUT_LEN) -> List[int]:
    ids = [INPUT_VOCAB[ch] for ch in expr]
    if len(ids) > max_len:
        raise ValueError(f"Input too long: {expr}")
    return ids + [INPUT_VOCAB["<pad>"]] * (max_len - len(ids))


def encode_multilabel_target(total: int) -> List[int]:
    return [int(ch) for ch in f"{total:04d}"]


def encode_seq_target(total: int, reverse: bool = False) -> Tuple[List[int], List[int]]:
    digits = list(f"{total:04d}")
    if reverse:
        digits = list(reversed(digits))
    dec_in = [OUTPUT_VOCAB["<sos>"]] + [OUTPUT_VOCAB[d] for d in digits]
    dec_out = [OUTPUT_VOCAB[d] for d in digits] + [OUTPUT_VOCAB["<eos>"]]
    return dec_in, dec_out


def decode_seq(ids: List[int]) -> str:
    toks = []
    for idx in ids:
        tok = OUTPUT_ID2TOK.get(int(idx), "")
        if tok in {"<pad>", "<sos>"}:
            continue
        if tok == "<eos>":
            break
        toks.append(tok)
    return "".join(toks)


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for dataset/dataloader usage. Please install torch.")


class AdditionClassificationDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]]):
        _require_torch()
        self.x = [encode_input(expr) for expr, _ in samples]
        self.y = [int(v) for _, v in samples]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


class AdditionRegressionDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], target_scale: float = 1998.0):
        _require_torch()
        self.x = [encode_input(expr) for expr, _ in samples]
        self.y = [float(v) / target_scale for _, v in samples]
        self.raw_y = [int(v) for _, v in samples]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.raw_y[idx], dtype=torch.long),
        )


class AdditionMultiLabelDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]]):
        _require_torch()
        self.x = [encode_input(expr) for expr, _ in samples]
        self.y = [encode_multilabel_target(v) for _, v in samples]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


class AdditionSeq2SeqDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], reverse: bool = False):
        _require_torch()
        self.src = [encode_input(expr) for expr, _ in samples]
        encoded = [encode_seq_target(v, reverse=reverse) for _, v in samples]
        self.tgt_in = [x[0] for x in encoded]
        self.tgt_out = [x[1] for x in encoded]
        self.raw_targets = [f"{v:04d}" for _, v in samples]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.src[idx], dtype=torch.long),
            torch.tensor(self.tgt_in[idx], dtype=torch.long),
            torch.tensor(self.tgt_out[idx], dtype=torch.long),
            self.raw_targets[idx],
        )


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    _require_torch()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
