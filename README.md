# DiffCF — Diffusion-based Counterfactuals for Time Series

This repository contains the runnable implementation for "Generating Realistic Time-Series Counterfactuals via Diffusion-Guided Sampling," accepted by ECML PKDD 2026.

High-level workflow:

1. Train a classifier on a dataset (`train_classifier.py`)
2. Train a diffusion model on the training split (`train_diffusion.py`)
3. Generate counterfactuals for the test split (`run_generate_cf.py`)
4. Evaluate counterfactual quality and save metrics/plots (`run_evaluate.py`)

---

## Installation

Create an environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Notes:

* **PyTorch** is required (`torch`). Install a CUDA build if you want GPU acceleration.

---

## How to run

All commands below assume you run them from the repo root.

### 1) Train classifier

```bash
python train_classifier.py --config configs/coffee.yaml
```

### 2) Train diffusion

```bash
python train_diffusion.py --config configs/coffee.yaml
```

### 3) Generate counterfactuals

This loads `classifier.pt` and `diffusion.pt` from `output.load_root/run_name/` and writes results to `output.root/run_name/`.

```bash
python run_generate_cf.py --config configs/coffee.yaml
```

### 4) Evaluate

```bash
python run_evaluate.py --config configs/coffee.yaml
```

### One-command pipeline (recommended)

`run_pipeline.py` will:

* train classifier if missing
* train diffusion if missing
* generate counterfactuals
* evaluate

```bash
python run_pipeline.py --config configs/coffee.yaml
```

---

## Multi-run helpers

There are a few convenience scripts to run multiple datasets / hyperparameters:

* `run_multi.py`
* `run_multi_parallel.py`
* `run_multi_grid_search.py`
* `grid_search.py`

The config `configs/multi_ucr.yaml` provides a template that:

* starts from a `base_config`
* iterates `datasets:` (different UCR datasets)
---
