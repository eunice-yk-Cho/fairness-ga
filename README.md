# Fairness-GA

A genetic algorithm tool for discovering fairness violations in ML classifiers.
It compares GA against random search under matched evaluation budgets.

## Requirements

- **Python** 3.8+
- Packages listed in `requirements.txt`

## Setup

### 1. Create a virtual environment (pick one)

**Conda:**
```bash
conda create -n fairness-ga python=3.10 -y
conda activate fairness-ga
```

**venv:**
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate.bat     # Windows CMD
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

```bash
python scripts/download_adult.py
python scripts/download_benchmark_datasets.py
```

This creates `data/adult.csv`, `data/compas.csv`, and `data/german_credit.csv`.

### 4. Run the experiment

```bash
python experiments/run_experiment.py
```

Default settings: 30 trials, 50,000 evaluations per method (set in `src/config.py`).

**Quick test (single dataset):**
```bash
python experiments/run_experiment.py --datasets adult --trials 3 --budget 5000
```

**All three datasets in parallel** (safe — results are merged atomically):
```bash
python experiments/run_experiment.py --datasets adult --trials 30 --budget 50000 --with-sensitivity &
python experiments/run_experiment.py --datasets compas --trials 30 --budget 50000 --with-sensitivity &
python experiments/run_experiment.py --datasets german --trials 30 --budget 50000 --with-sensitivity &
wait
```

Each process writes only its own dataset into `combined_results.json` without
overwriting the others.

**With hyperparameter sensitivity:**
```bash
python experiments/run_experiment.py --with-sensitivity --sensitivity-trials 3
```

Outputs are saved under `experiments/results/`.

### 5. Statistical test

```bash
python experiments/statistical_test.py
```

Runs a one-sided Wilcoxon signed-rank test (GA > Random) for each dataset and metric.

### 6. Visualization

```bash
python experiments/visualization.py
python experiments/visualization.py --dataset compas
```

Saves figures to `experiments/fairness_ga_results_<dataset>.png`.

## Quick start (conda, all-in-one)

```bash
conda create -n fairness-ga python=3.10 -y && conda activate fairness-ga
pip install -r requirements.txt
python scripts/download_adult.py
python scripts/download_benchmark_datasets.py
python experiments/run_experiment.py
python experiments/statistical_test.py
python experiments/visualization.py
```