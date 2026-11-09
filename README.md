# Fairness-GA

Can a genetic algorithm find more fairness violations in a machine-learning
classifier than plain random guessing — given the same number of tries?

That's the whole question this project tries to answer. The short version of the
answer, after 30 trials across three benchmark datasets, is: yes, and usually by
a wide margin.

## The idea

When a model is used for something that matters — approving a loan, screening a
job application, estimating whether someone will re-offend — we'd like to know
whether it treats people differently just because of a protected attribute like
sex or race. A natural way to probe this is **counterfactual testing**: take an
input, flip only its sensitive attribute, and ask the model to score both
versions. If the two predictions disagree a lot, you've found an input the model
treats unfairly.

The catch is that the input space is huge, and the unfair regions are often
small and hidden. You can't test every possible input, so you need a search that
spends its limited budget wisely. This repository pits two such searches against
each other under a **matched evaluation budget**:

- **Genetic algorithm (GA)** — evolves a population of inputs, keeping and
  recombining the ones that expose the largest prediction gap.
- **Random search (RS)** — the honest baseline. Draw inputs uniformly at random
  and keep score.

Both get the exact same number of model evaluations, so any difference comes
from the search strategy, not from one method simply being allowed to look more.

### How "unfairness" is measured

For a single input `x`, the **discrimination score** is

```
| P(positive | x) − P(positive | flip_sensitive(x)) |
```

i.e. how much the predicted probability moves when you toggle the protected
attribute and change nothing else. The GA's fitness function is exactly this
score — it is literally rewarded for finding inputs the model judges unfairly.

On top of the per-input score, each run also reports a few aggregate fairness
metrics so the results aren't resting on a single definition:

- **Individual count** — how many generated inputs cross a discrimination
  threshold (default `0.2`).
- **Individual mean** — the average discrimination score across all evaluations.
- **Demographic parity difference** — gap in positive-prediction rates between
  the two groups.
- **Equalized odds difference (proxy)** — the same idea, but conditioned on the
  (counterfactual) label.

## What's in the box

```
src/
  config.py            # all the knobs: seeds, population size, budget, datasets
  preprocessing.py     # load CSV, encode columns, binarise the sensitive attribute
  model.py             # the classifier under test (RandomForest)
  fairness.py          # flip_sensitive() + discrimination_score()
  genetic_algorithm.py # the GA: tournament selection, crossover, mutation, elitism
  random_search.py     # the matched-budget baseline
  metrics.py           # search efficiency + group-fairness metrics
  utils.py             # feature-range sampling helpers

experiments/
  run_experiment.py    # the main driver — trains models and runs both searches
  statistical_test.py  # paired Wilcoxon test, GA vs RS
  visualization.py     # convergence curves + metric plots
  results/             # saved metrics, convergence arrays, combined_results.json

scripts/
  download_adult.py              # fetch the Adult dataset from UCI
  download_benchmark_datasets.py # fetch COMPAS and German Credit

data/                  # CSVs land here after the download scripts run
```

A couple of implementation details worth calling out, because they're easy to
get wrong:

- The GA uses **elitism** — the best individual always survives to the next
  generation, so progress never goes backwards.
- Fitness is evaluated in **batches** (`predict_proba` on the whole population at
  once), which is what makes a 50,000-evaluation budget tolerable to run.
- Sensitive attributes are normalised to `{0, 1}` with some care (male/female
  tokens, binary numerics, alphabetical fallback with a warning) so flipping
  them is always well-defined.

## Getting started

### 1. Environment

Python 3.8+ . Pick whichever you prefer:

```bash
# conda
conda create -n fairness-ga python=3.10 -y
conda activate fairness-ga

# …or venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.bat
```

### 2. Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

(Versions known to work: `numpy<2.0`, `pandas<3.1`, `scikit-learn<2.0`,
`scipy<2.0`, `matplotlib<4.0`.)

### 3. Get the datasets

```bash
python scripts/download_adult.py
python scripts/download_benchmark_datasets.py
```

This writes `data/adult.csv`, `data/compas.csv`, and `data/german_credit.csv`,
pulled from UCI and ProPublica. The downloads are idempotent — they skip files
that already exist.

| Dataset       | Target              | Sensitive attribute |
| ------------- | ------------------- | ------------------- |
| Adult         | income `>50K`       | sex                 |
| COMPAS        | two-year recidivism | race                |
| German Credit | credit `good`       | sex                 |

### 4. Run the experiment

```bash
python experiments/run_experiment.py
```

Defaults (all live in `src/config.py`): 30 trials per dataset, 50,000 model
evaluations per method per trial, RandomForest classifier, fixed seed `42`.

Want a quick sanity check instead of the full run?

```bash
python experiments/run_experiment.py --datasets adult --trials 3 --budget 5000
```

The three datasets are independent, and results are merged into
`combined_results.json` atomically (via a lock file), so it's safe to run them in
parallel:

```bash
python experiments/run_experiment.py --datasets adult  --with-sensitivity &
python experiments/run_experiment.py --datasets compas --with-sensitivity &
python experiments/run_experiment.py --datasets german --with-sensitivity &
wait
```

The `--with-sensitivity` flag adds a hyperparameter sweep over population size
and mutation rate, so you can see how robust the GA is to its own settings.

### 5. Test for significance

```bash
python experiments/statistical_test.py
```

This runs a one-sided **Wilcoxon signed-rank test** (the paired,
non-parametric one — appropriate because the same trial seeds are used for both
methods) asking whether GA genuinely beats RS, per dataset and per metric.
Output is printed and saved to `experiments/stat_test_summary.txt`.

### 6. Make the plots

```bash
python experiments/visualization.py --dataset adult
```

Produces convergence curves (GA vs RS over the budget) and the per-trial metric
comparisons.

## What the results say

Across all three datasets, the GA finds dramatically more discriminatory inputs
than random search at the same budget, and the gap is statistically significant.
A few representative numbers from a 30-trial run (`stat_test_summary.txt`):

| Dataset | Metric          | GA (mean)  | RS (mean) | GA wins | p-value   |
| ------- | --------------- | ---------- | --------- | ------- | --------- |
| COMPAS  | individual_count| 48,986     | 278       | 100 %   | 9.3e-10   |
| Adult   | individual_count| 47,075     | 4.7       | 96.7 %  | 1.9e-09   |
| German  | individual_mean | 0.169      | 0.027     | 100 %   | 9.3e-10   |

The "individual_count" rows are the headline: random search barely stumbles onto
inputs that cross the discrimination threshold, while the GA finds them by the
tens of thousands. The group-fairness metrics (demographic parity, equalized
odds) tell the same story, though with more variance — searching for the
*worst* inputs naturally produces noisier group statistics than measuring a
fixed test set does.

It's worth being honest about what this does and doesn't show. The GA is an
excellent *bug finder*: it efficiently surfaces inputs where the model behaves
unfairly. That's not the same as saying the model is unfair on realistic data —
the test set's own fairness metrics (also recorded in `combined_results.json`)
are far milder. The point is that the violations *exist and are findable*, and a
guided search finds them far more efficiently than blind sampling.

## Reproducibility notes

- Every trial seeds `numpy` and `random` from a deterministic offset of
  `RANDOM_SEED`, so a given config reproduces exactly.
- GA and RS share trial seeds, which is what makes the *paired* statistical test
  valid.
- All tunables — budget, population size, mutation rate, datasets, thresholds —
  live in `src/config.py` and are overridable from the command line.

## Documentation

Longer write-ups (the full report, replication guide, and user manual) are
included as PDFs in the repository root: `report.pdf`, `replication.pdf`, and
`manual.pdf`.

---

*Built as a search-based software engineering study on AI model fairness
testing. Author: Yoonkyoung Cho.*
