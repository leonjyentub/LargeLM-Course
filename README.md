# NLP Addition Learning (PyTorch)

This project trains models to learn string-based addition from data only.
No rule-based arithmetic is provided to the model.

## Data
- Input: expression string such as `"100+20"`
- Target:
  - Class A/C: single class id in `[0, 1999]`
  - Regression: normalized scalar `sum / 1998`
  - Class B: 4-digit decomposition (`0-9` for each position)
  - Class D: generated output sequence, left-to-right or right-to-left

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
6. `train_lstm_regression.py` (LSTM regression with MSE loss)
7. `train_lstm_seq2seq.py` (LSTM Seq2Seq, left-to-right generation)
8. `train_lstm_reverse_seq2seq.py` (LSTM Seq2Seq, right-to-left generation)
9. `train_transformer_seq2seq.py` (Transformer Seq2Seq)

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

Run all configured methods + comparison using config:

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

## Final Experiment Results

Full experiment ranking:

| Rank | Method | Main results |
|---:|---|---|
| 1 | `ffnn` | accuracy=0.5516, macro_f1=0.4502, auc=0.9938 |
| 2 | `gru_cls` | accuracy=0.0565, macro_f1=0.0217, auc=0.9655 |
| 3 | `lstm_cls` | accuracy=0.0449, macro_f1=0.0160, auc=0.9602 |
| 4 | `self_attention` | accuracy=0.0158, macro_f1=0.0050, auc=0.9076 |
| 5 | `lstm_reverse_seq2seq` | exact_match=0.9025, char_accuracy=0.9700 |
| 6 | `transformer_seq2seq` | exact_match=0.8930, char_accuracy=0.9610 |
| 7 | `lstm_seq2seq` | exact_match=0.1356, char_accuracy=0.6751 |
| 8 | `lstm_regression` | exact_match=0.0615, mae=5.27, rmse=7.04 |
| 9 | `lstm_multilabel` | exact_match=0.0195 |

Teaching notes:
- Multiclass methods are ranked by classification metrics (`accuracy`, `macro_f1`, sampled multiclass `auc`).
- Seq2Seq methods are ranked by full-answer `exact_match` and per-character `char_accuracy`.
- Regression reports rounded-answer `exact_match` plus numeric error (`mae`, `rmse`).
- The right-to-left LSTM Seq2Seq result is a useful classroom contrast: generating the least-significant digit first aligns better with addition carry direction than the left-to-right LSTM Seq2Seq baseline.

## Shared Modules
- `common/data_utils.py`: dataset generation, encoding, Dataset classes
- `common/models.py`: all model architectures
- `common/train_eval.py`: training/evaluation loop, metrics, ROC/AUC
- `common/plot_utils.py`: seaborn plotting helpers
- `common/logging_utils.py`: unified logger
- `common/run_helpers.py`: reusable loader/data preparation wrappers

## Teaching Slides

Editable course deck:

```bash
node slides/build_addition_course_deck.mjs
```

Output:
- `slides/output/addition_nlp_course.pptx`

## Dependencies

```bash
pip install -r requirements.txt
```
