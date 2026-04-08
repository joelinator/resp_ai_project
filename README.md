# Final Project Pipeline

This repository now contains a complete end-to-end implementation of the project described in `homework_instructions.md` and `myIdea.md`.

## Run

Use the requested virtual environment Python:

```bash
/home/joelinator/env/bin/python run_project.py
```

Reframed 5x simulation run (used by latest report/presentation):

```bash
/home/joelinator/env/bin/python run_project.py --simulation-timestep-multiplier 5 --results-dir results_reframed_proficiency_5x
```

## What the script does

1. Downloads ASSISTments 2009-2010 data if missing.
2. Preprocesses and creates deterministic synthetic gender.
3. Trains a calibrated KT model (RandomForest + Platt scaling).
4. Simulates heterogeneous teachers.
5. Evaluates:
   - baseline confidence-threshold policy,
   - proposed LinUCB routing with fairness guardrail.
6. Logs metrics, fairness tables, policy traces, and plots to `results/`.
7. Generates transparency artifacts (global + local SHAP plots).
8. Copies NeurIPS style/checklist assets into `paper/`.

## Outputs

- `results/metrics_summary.csv`
- `results/fairness_metrics.csv`
- `results/model_metrics.json`
- `results/risk_coverage_*.png`
- `results/shap_*.png`
- `results/baseline_policy_logs.csv`
- `results/proposed_policy_logs.csv`
- `results/guardrail_history.csv`
- `results/analysis.md`
- `paper/paper.tex` and `paper/references.bib`

## Overleaf (recommended)

For easiest compilation on Overleaf, upload the whole repository and set the main file to:

- `main.tex`

Why:
- `main.tex` is root-based (no parent `..` image paths),
- it references all needed assets directly from `results/`, `images/`, and `paper/`,
- it is robust to either flowchart filename:
  - `images/proposed_vs_baseline_flowchart.png`, or
  - `images/proposed_vs_baseline_flowchart.txt.png`.
