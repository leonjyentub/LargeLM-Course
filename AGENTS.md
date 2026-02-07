# CODEX Experiment Guide

Base policy: `AI_EXPERIMENT_PLAYBOOK.md`

## Role
- Act as an implementation-first coding agent.
- Prefer editing files directly and running validation commands.
- Avoid long planning-only answers when code changes are requested.

## Working Style
- Keep responses concise, concrete, and path-specific.
- Reuse modules under `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/`.
- Do not duplicate training/data/eval logic unless required.

## Must-Do for New Methods
1. Add model class in `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/common/models.py`.
2. Add standalone `train_<method>.py` script.
3. Update both config files under `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/configs/`.
4. Update `/Users/leonjye/Documents/PythonProjects/LargeLM-Course/compare_all_methods.py`.
5. Ensure outputs follow `outputs/<method>/` contract.
6. Run compile and dry-run checks.

## Reporting Format
- List changed files with absolute paths.
- Include exact run commands.
- If blocked, state blocker and next required input.
