# NLP Addition Learning (PyTorch)

This project trains models to learn string-based addition from data only.
No rule-based arithmetic is provided to the model.

## Data
- Input: expression string such as `"100+20"`
- Target:
  - Class A/C: single class id in `[0, 1999]`
  - Class B: 4-digit decomposition (`0-9` for each position)
  - Class D: generated output sequence

Generate dataset first:

```bash
python3 generate_data.py --data-dir data/addition --n-train 80000 --n-val 10000 --n-test 10000
```

## Methods and Scripts
1. `train_ffnn_classifier.py` (Simple FFNN)
2. `train_lstm_classifier.py` (LSTM encoder classifier)
3. `train_gru_classifier.py` (GRU encoder classifier)
4. `train_lstm_multilabel.py` (LSTM multi-label digit classifier)
5. `train_self_attention_classifier.py` (Self-attention classifier)
6. `train_lstm_seq2seq.py` (LSTM Seq2Seq)
7. `train_transformer_seq2seq.py` (Transformer Seq2Seq)

Each script outputs to `outputs/<method>/`:
- `train.log`
- `metrics.json`
- `training_curves.png`
- `best_model.pt`
- `roc_curve.png` (for multiclass methods)

## One-Click Full Run (New)

Fixed config files:
- JSON: `configs/experiment_config.json`
- YAML: `configs/experiment_config.yaml`

Run all 7 methods + comparison using config:

```bash
python3 run_all_experiments.py --config configs/experiment_config.json
```

Dry-run (print commands only):

```bash
python3 run_all_experiments.py --config configs/experiment_config.json --dry-run
```

If using YAML config, install parser first:

```bash
pip install pyyaml
```

## Compare All Methods

```bash
python3 compare_all_methods.py --run --epochs 8 --batch-size 256 --lr 1e-3
```

Comparison outputs:
- `outputs/comparison/all_methods_metrics.csv`
- `outputs/comparison/all_methods_comparison.png`

## Shared Modules
- `common/data_utils.py`: dataset generation, encoding, Dataset classes
- `common/models.py`: all model architectures
- `common/train_eval.py`: training/evaluation loop, metrics, ROC/AUC
- `common/plot_utils.py`: seaborn plotting helpers
- `common/logging_utils.py`: unified logger
- `common/run_helpers.py`: reusable loader/data preparation wrappers

## Dependencies

```bash
pip install -r requirements.txt
```
