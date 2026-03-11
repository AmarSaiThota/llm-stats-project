# How Sampling Choices and Probability Calibration Affect LLM Outputs

An empirical study on GPT-2 (124M) of how decoding parameters (temperature, top-p, top-k) affect the **correctness**, **calibration**, and **diversity** of model outputs. Full write-up: [report/final_report.md](report/final_report.md).

## Project timeline

Developed over about 10 weeks, January–March 2026, with weekly advisor meetings. The work went in this order:

1. Text generation with configurable sampling (`generate.py`)
2. Zero-shot multiple-choice evaluation on ARC-Easy (`eval_mcq.py`)
3. Calibration metrics: ECE, Brier, NLL, reliability and selective-accuracy plots (`calibration.py`)
4. Temperature × top-p × top-k grid sweep (`sweep.py`)
5. Accuracy vs. calibration tradeoff plots (`plot_tradeoffs.py`)
6. Post-hoc temperature scaling (`temp_scaling.py`)
7. Robustness check on SciQ (`robustness.py`)
8. Generation diversity: Distinct-n, Self-BLEU, embedding dispersion (`diversity.py`)
9. Findings notes and final report

## Layout

- `scripts/`: experiment code
- `results/`: CSV outputs (`results/sweep/` holds per-configuration results)
- `figures/`: plots used in the report
- `report/`: final report
