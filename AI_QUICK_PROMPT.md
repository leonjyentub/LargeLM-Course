# AI Quick Prompt

Use this prompt for fast, consistent model extension requests.

```text
Follow: /Users/leonjye/Documents/PythonProjects/LargeLM-Course/AI_EXPERIMENT_PLAYBOOK.md

Task: Add one new NLP model/training method to this addition-learning project.
Constraints:
1) No arithmetic rules in model logic; learn only from data.
2) Reuse shared modules under /common.
3) Create standalone script: train_<method>.py with standard args.
4) Output contract must be:
   outputs/<method>/train.log
   outputs/<method>/metrics.json
   outputs/<method>/training_curves.png
   outputs/<method>/best_model.pt
5) Update both config files under /configs.
6) Update compare_all_methods.py so this method is included in comparison.
7) Run checks:
   python3 -m compileall common *.py
   python3 run_all_experiments.py --config configs/experiment_config.json --dry-run
8) Return:
   - changed files (absolute paths)
   - exact run commands
   - any limitations/blockers
```
