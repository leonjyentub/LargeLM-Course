# ANTIGRAVITY Experiment Guide

Base policy: `AI_EXPERIMENT_PLAYBOOK.md`

## Role
- Act as a fast iteration coding assistant with strict project guardrails.
- Focus on reproducible experiments and minimal-friction execution.

## Working Style
- Keep edits small, testable, and modular.
- Follow existing naming and output conventions exactly.
- Do not bypass shared modules when a reusable path already exists.

## Must-Do for New Methods
1. Implement model in shared model module.
2. Create `train_<method>.py` with standard CLI args.
3. Update config registry and comparison registry.
4. Confirm output artifacts are generated under `outputs/<method>/`.
5. Run compile + dry-run checks before finalizing.

## Reporting Format
- Return concise change log with absolute paths.
- Include one-command run examples for smoke/full training.
