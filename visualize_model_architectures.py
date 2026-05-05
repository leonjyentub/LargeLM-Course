from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torchview import draw_graph

from common.data_utils import (
    INPUT_VOCAB,
    MAX_INPUT_LEN,
    OUTPUT_VOCAB,
    encode_input,
    encode_seq_target,
    load_splits,
)
from common.models import (
    FFNNClassifier,
    GRUClassifier,
    LSTMClassifier,
    LSTMMultiLabelClassifier,
    LSTMRegressor,
    LSTMReverseSeq2Seq,
    LSTMSeq2Seq,
    SelfAttentionClassifier,
    TransformerSeq2Seq,
)
from common.run_helpers import vocab_sizes


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    build_model: Callable[[], nn.Module]
    build_inputs: Callable[[list[tuple[str, int]], torch.device], tuple[torch.Tensor, ...]]


def _total_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _classification_inputs(samples: list[tuple[str, int]], device: torch.device) -> tuple[torch.Tensor]:
    expr = samples[0][0] if samples else "123+45"
    x = torch.tensor([encode_input(expr)], dtype=torch.long, device=device)
    return (x,)


def _seq2seq_inputs(samples: list[tuple[str, int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    expr, total = samples[0] if samples else ("123+45", 168)
    src = torch.tensor([encode_input(expr)], dtype=torch.long, device=device)
    tgt_in, _ = encode_seq_target(total)
    tgt = torch.tensor([tgt_in], dtype=torch.long, device=device)
    return src, tgt


def _build_specs() -> list[ModelSpec]:
    src_vocab_size, tgt_vocab_size = vocab_sizes()
    input_vocab_size = len(INPUT_VOCAB)

    return [
        ModelSpec("ffnn_classifier", "FFNN Classifier", lambda: FFNNClassifier(vocab_size=input_vocab_size), _classification_inputs),
        ModelSpec("lstm_classifier", "LSTM Classifier", lambda: LSTMClassifier(vocab_size=input_vocab_size), _classification_inputs),
        ModelSpec("gru_classifier", "GRU Classifier", lambda: GRUClassifier(vocab_size=input_vocab_size), _classification_inputs),
        ModelSpec("lstm_regression", "LSTM Regression", lambda: LSTMRegressor(vocab_size=input_vocab_size), _classification_inputs),
        ModelSpec(
            "lstm_multilabel",
            "LSTM Multi-Label Classifier",
            lambda: LSTMMultiLabelClassifier(vocab_size=input_vocab_size),
            _classification_inputs,
        ),
        ModelSpec(
            "self_attention_classifier",
            "Self-Attention Classifier",
            lambda: SelfAttentionClassifier(vocab_size=input_vocab_size),
            _classification_inputs,
        ),
        ModelSpec(
            "lstm_seq2seq",
            "LSTM Seq2Seq",
            lambda: LSTMSeq2Seq(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size),
            _seq2seq_inputs,
        ),
        ModelSpec(
            "lstm_reverse_seq2seq",
            "LSTM Reverse Seq2Seq",
            lambda: LSTMReverseSeq2Seq(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size),
            _seq2seq_inputs,
        ),
        ModelSpec(
            "transformer_seq2seq",
            "Transformer Seq2Seq",
            lambda: TransformerSeq2Seq(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size),
            _seq2seq_inputs,
        ),
    ]


def _render_torchview_jpg(
    *,
    spec: ModelSpec,
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    out_dir: Path,
    depth: int,
) -> Path:
    title = (
        f"{spec.display_name}\\n"
        f"total params: {_total_params(model):,} | trainable params: {_trainable_params(model):,}"
    )
    graph = draw_graph(
        model,
        input_data=inputs,
        graph_name=title,
        depth=depth,
        device="cpu",
        mode="eval",
        strict=False,
        expand_nested=True,
        show_shapes=True,
        save_graph=False,
    )
    graph.visual_graph.attr(rankdir="TB")
    graph.visual_graph.attr("node", fontsize="10")

    rendered_path = graph.visual_graph.render(
        filename=spec.name,
        directory=str(out_dir),
        format="jpg",
        cleanup=True,
    )
    return Path(rendered_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize every PyTorch model architecture as JPG files with torchview.")
    parser.add_argument("--data-dir", type=str, default="data/addition", help="Directory containing train/val/test CSV files.")
    parser.add_argument("--out-dir", type=str, default="outputs/model_visualizations", help="Directory for JPG outputs.")
    parser.add_argument("--depth", type=int, default=4, help="torchview graph nesting depth.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples, _val_samples, _test_samples = load_splits(data_dir)
    samples = [next((sample for sample in train_samples if len(sample[0]) == MAX_INPUT_LEN), train_samples[0])]
    device = torch.device("cpu")

    warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage.*")

    print(f"Using data sample from {data_dir / 'train.csv'}")
    print(f"Input vocab: {len(INPUT_VOCAB)} tokens, output vocab: {len(OUTPUT_VOCAB)} tokens")
    print(f"Max input length: {MAX_INPUT_LEN}")

    for spec in _build_specs():
        model = spec.build_model().to(device).eval()
        inputs = spec.build_inputs(samples, device)
        out_path = _render_torchview_jpg(
            spec=spec,
            model=model,
            inputs=inputs,
            out_dir=out_dir,
            depth=args.depth,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
