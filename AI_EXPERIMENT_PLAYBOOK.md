# AI Experiment Playbook (NLP Addition Project)

This document is the **single source of truth** for any AI agent (or developer) that extends this project with new models, training methods, or comparison outputs.

## 1. Project Goal and Hard Constraints

- Task: Learn mapping from string expression to result, e.g. `"100+20" -> "120"`.
- Scope: Two positive integers in `[1, 999]`.
- Learning principle: **No rule-based arithmetic logic inside models**.
- Allowed dependencies: `pytorch`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`.
- Data generation must remain data-driven and reproducible by seed.

## 2. Existing Architecture

- Data utilities: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/data_utils.py`
- Model definitions: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/models.py`
- Training/eval loops: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/train_eval.py`
- Plotting: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/plot_utils.py`
- Logging: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/logging_utils.py`
- Runners/helpers: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/run_helpers.py`
- One-click orchestrator: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/run_all_experiments.py`
- Comparison script: `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/compare_all_methods.py`
- Configs:
  - `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/configs/experiment_config.json`
  - `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/configs/experiment_config.yaml`

## 3. Standard Output Contract (Must Follow)

Every training script must write to `outputs/<method_name>/`:

- `train.log`
- `metrics.json`
- `training_curves.png`
- `best_model.pt`
- `roc_curve.png` (if multiclass ROC/AUC is applicable)

`metrics.json` should include:

- Always: `task_type`, `test_loss`
- For multiclass: `accuracy`, `macro_f1`, optional `auc`
- For multilabel: `exact_match`, `digit_accuracy`
- For seq2seq: `exact_match`, `char_accuracy`

## 4. How to Add a New Model (Checklist)

1. Define model class in `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/models.py`.
2. Reuse existing dataset pipeline from `common/data_utils.py` and `common/run_helpers.py`.
3. Decide task type:
   - `multiclass`, `multilabel`, or `seq2seq`.
4. Reuse correct trainer:
   - `run_classifier_training(...)` or `run_seq2seq_training(...)`.
5. Create standalone training script with English filename:
   - `train_<new_method_name>.py`.
6. Add method entry into:
   - `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/configs/experiment_config.json`
   - `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/configs/experiment_config.yaml`
7. Add method mapping in `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/compare_all_methods.py`.
8. Run a smoke test (small data, 1 epoch).
9. Run full training and verify outputs are generated.

## 5. Naming Rules

- Script filename: `train_<method>.py`
- Output directory: `outputs/<method>/`
- Method key in comparison/config should match output folder name.
- Keep names stable to avoid breaking aggregated comparison.

## 6. Fair Comparison Rules

To claim method comparison is valid:

- Same train/val/test split (same seed and dataset files).
- Same base training budget when possible (`epochs`, `batch_size`, `lr`).
- Any special override must be recorded in config (`overrides`).
- Compare by the right metric family:
  - Classification: `accuracy`, `macro_f1`, optional `auc`
  - Generation: `exact_match`, `char_accuracy`

## 7. Minimum Quality Gate Before Commit

- Code compiles:
  - `python3 -m compileall common *.py`
- Config dry-run works:
  - `python3 run_all_experiments.py --config configs/experiment_config.json --dry-run`
- New script can parse CLI args and start without import errors.
- `metrics.json` keys follow Section 3.

## 8. Suggested CLI Template for New Training Script

Use this argument style for consistency:

- `--data-dir`
- `--out-dir`
- `--epochs`
- `--batch-size`
- `--lr`
- `--seed`
- `--regen-data` (flag)

## 9. AI Prompt Template (Copy and Reuse)

Use this prompt when asking AI to extend the project:

```text
Please add one new NLP method to this addition-learning project.
Constraints:
1) No arithmetic rules in model logic; must learn only from data.
2) Reuse common modules in /common instead of duplicating code.
3) Create standalone script train_<method>.py with standard CLI args.
4) Ensure outputs follow outputs/<method>/ contract:
   train.log, metrics.json, training_curves.png, best_model.pt.
5) Update compare_all_methods.py and both config files under /configs.
6) Run compile check and provide exact changed file paths.
```

## 10. Common Mistakes to Avoid

- Do not hardcode addition/carry rules in preprocessing or model forward pass.
- Do not create a one-off data split that breaks comparability.
- Do not skip config/comparison updates when adding a new method.
- Do not change output metric key names without updating comparator.
- Do not introduce extra dependencies unless necessary.

## 11. Recommended Workflow

1. Edit model + script.
2. Update configs + comparison registry.
3. Run smoke test.
4. Run full experiment.
5. Inspect `outputs/comparison/all_methods_metrics.csv` and chart.
6. Commit with clear message: `add <method> experiment pipeline`.

---

If future requirements change (new metric, new task formulation), update this file first, then implement code changes.
