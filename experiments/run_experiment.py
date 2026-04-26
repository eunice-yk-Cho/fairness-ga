# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import argparse
import json
import os
import time
from types import SimpleNamespace

# Add project root to path so that `src` is importable when running from repo root
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from sklearn.metrics import accuracy_score
from src import config
from src.preprocessing import load_and_preprocess
from src.model import train_model
from src.utils import compute_feature_ranges, make_sampler
from src.random_search import random_search
from src.genetic_algorithm import GeneticAlgorithm
from src.metrics import (
    search_efficiency,
    demographic_parity_difference,
    equalized_odds_difference_proxy,
)

def _merge_into_combined(combined_path, dataset_key, dataset_entry, config_entry):
    """Safely merge results into combined_results.json using file locking."""
    lock_path = combined_path.with_suffix(".lock")
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    try:
        if combined_path.exists():
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
        else:
            combined = {"datasets": {}, "config": config_entry}
        combined["datasets"][dataset_key] = dataset_entry
        combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        os.unlink(str(lock_path))


def _pad_curves(curve_list):
    """Pad convergence curves to equal length."""
    max_len = max(len(c) for c in curve_list)
    return [c + [c[-1]] * (max_len - len(c)) for c in curve_list]


def _parse_csv_list(raw, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]

def _ga_config(population_size, mutation_rate):
    return SimpleNamespace(
        POPULATION_SIZE=population_size,
        TOURNAMENT_K=config.TOURNAMENT_K,
        MUTATION_RATE=mutation_rate,
    )

def _trial_metrics(model, details, sensitive_index):
    scores = np.asarray(details["scores"])
    samples = np.asarray(details["samples"])
    return {
        "individual_count": int(search_efficiency(scores, config.DISCRIMINATION_THRESHOLD)),
        "individual_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "demographic_parity": demographic_parity_difference(model, samples, sensitive_index),
        "equalized_odds": equalized_odds_difference_proxy(model, samples, sensitive_index),
    }

def _run_ga_trial(model, sampler, sensitive_index, feature_ranges, budget, ga_conf,
                   categorical_indices=frozenset()):
    ga = GeneticAlgorithm(
        model=model,
        sampler=sampler,
        sensitive_index=sensitive_index,
        config=ga_conf,
        feature_ranges=feature_ranges,
        categorical_indices=categorical_indices,
    )
    return ga.run(n_evals=budget, return_details=True)

def _run_sensitivity(model, sampler, sensitive_index, feature_ranges, budget, trials, pop_sizes, mutation_rates,
                     categorical_indices=frozenset()):
    records = []
    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            ga_conf = _ga_config(pop_size, mutation_rate)
            trial_counts = []
            trial_means = []
            for t in range(trials):
                trial_seed = config.RANDOM_SEED + 10_000 + t * 7 + pop_size * 131 + int(mutation_rate * 10000) * 37
                config.set_seed(trial_seed)
                details = _run_ga_trial(model, sampler, sensitive_index, feature_ranges, budget, ga_conf,
                                        categorical_indices)
                trial_counts.append(int(search_efficiency(details["scores"], config.DISCRIMINATION_THRESHOLD)))
                trial_means.append(float(np.mean(details["scores"])))
            records.append({
                "population_size": int(pop_size),
                "mutation_rate": float(mutation_rate),
                "mean_discriminatory_cases": float(np.mean(trial_counts)),
                "std_discriminatory_cases": float(np.std(trial_counts)),
                "mean_score": float(np.mean(trial_means)),
                "std_score": float(np.std(trial_means)),
            })
    return records

def main():
    parser = argparse.ArgumentParser(description="Run advanced GA fairness experiments.")
    parser.add_argument("--trials", type=int, default=config.N_TRIALS, help="Number of independent trials.")
    parser.add_argument("--budget", type=int, default=config.evaluation_budget(), help="Evaluations per method per trial.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(config.DEFAULT_DATASETS),
        help="Comma-separated dataset keys (adult, compas, german).",
    )
    parser.add_argument("--ga-population", type=int, default=config.POPULATION_SIZE, help="GA population size.")
    parser.add_argument("--ga-mutation", type=float, default=config.MUTATION_RATE, help="GA mutation rate.")
    parser.add_argument("--with-sensitivity", action="store_true", help="Run hyperparameter sensitivity analysis.")
    parser.add_argument("--sensitivity-trials", type=int, default=5, help="Trials per sensitivity setting.")
    parser.add_argument(
        "--sensitivity-pop-sizes",
        type=str,
        default=",".join(str(x) for x in config.SENSITIVITY_POPULATION_SIZES),
        help="Comma-separated population sizes.",
    )
    parser.add_argument(
        "--sensitivity-mutation-rates",
        type=str,
        default=",".join(str(x) for x in config.SENSITIVITY_MUTATION_RATES),
        help="Comma-separated mutation rates.",
    )
    args = parser.parse_args()

    config.set_seed()
    budget = args.budget
    n_trials = args.trials
    dataset_keys = _parse_csv_list(args.datasets, str)
    ga_conf = _ga_config(args.ga_population, args.ga_mutation)
    pop_sizes = _parse_csv_list(args.sensitivity_pop_sizes, int)
    mutation_rates = _parse_csv_list(args.sensitivity_mutation_rates, float)
    results_root = _root / "experiments" / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Starting experiment loop (datasets={dataset_keys}, trials={n_trials}, budget={budget})")
    for dataset_key in dataset_keys:
        if dataset_key not in config.DATASET_CONFIGS:
            print(f"  [Skip] Unknown dataset key: {dataset_key}")
            continue

        ds_cfg = config.DATASET_CONFIGS[dataset_key]
        ds_path = _root / ds_cfg["path"]
        if not ds_path.exists():
            print(f"  [Skip] Dataset file not found: {ds_path}")
            continue

        print(f"\n[2/5] Loading/preprocessing dataset={dataset_key} ...")
        X_train, X_test, y_train, y_test, sensitive_index, categorical_indices = load_and_preprocess(
            str(ds_path),
            ds_cfg["sensitive_attr"],
            ds_cfg["target"],
            ds_cfg["positive_label"],
        )
        print(f"      train={X_train.shape}, test={X_test.shape}, sensitive_index={sensitive_index}")

        print("[3/5] Training model ...")
        model = train_model(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"      Model accuracy - train: {train_acc:.4f}, test: {test_acc:.4f}")
        feature_ranges = compute_feature_ranges(X_train)
        sampler = make_sampler(feature_ranges, categorical_indices)

        # Compute real test-set fairness metrics
        y_pred_test = model.predict(X_test)
        groups = X_test[:, sensitive_index]
        g0_mask = groups == 0
        g1_mask = groups == 1
        real_dp = float(abs(np.mean(y_pred_test[g0_mask]) - np.mean(y_pred_test[g1_mask])))
        real_eo_diffs = []
        for true_label in (0, 1):
            true_mask = y_test == true_label
            if np.sum(true_mask & g0_mask) == 0 or np.sum(true_mask & g1_mask) == 0:
                continue
            rate0 = float(np.mean(y_pred_test[true_mask & g0_mask]))
            rate1 = float(np.mean(y_pred_test[true_mask & g1_mask]))
            real_eo_diffs.append(abs(rate0 - rate1))
        real_eo = float(max(real_eo_diffs) if real_eo_diffs else 0.0)
        real_test_fairness = {"demographic_parity": real_dp, "equalized_odds": real_eo}
        print(f"      Real test-set fairness - DP: {real_dp:.4f}, EO: {real_eo:.4f}")

        metrics = {
            "individual_count": {"ga": [], "rs": []},
            "individual_mean": {"ga": [], "rs": []},
            "demographic_parity": {"ga": [], "rs": []},
            "equalized_odds": {"ga": [], "rs": []},
        }
        curves = {"ga_best": [], "ga_mean": [], "rs_best": [], "rs_mean": []}

        print("[4/5] Running trials ...")
        for t in range(n_trials):
            trial_seed = config.RANDOM_SEED + t
            config.set_seed(trial_seed)
            print(f"      Trial {t + 1}/{n_trials} (seed={trial_seed})")

            ga_details = _run_ga_trial(model, sampler, sensitive_index, feature_ranges, budget, ga_conf,
                                        categorical_indices)
            config.set_seed(trial_seed + 10_000)
            rs_chunk = max(1, args.ga_population - 1)
            rs_details = random_search(
                model=model,
                sampler=sampler,
                sensitive_index=sensitive_index,
                n_iter=budget,
                chunk_size=rs_chunk,
                return_details=True,
            )
            ga_metric = _trial_metrics(model, ga_details, sensitive_index)
            rs_metric = _trial_metrics(model, rs_details, sensitive_index)

            for k in metrics.keys():
                metrics[k]["ga"].append(float(ga_metric[k]))
                metrics[k]["rs"].append(float(rs_metric[k]))

            curves["ga_best"].append(ga_details["best_curve"])
            curves["ga_mean"].append(ga_details["mean_curve"])
            curves["rs_best"].append(rs_details["best_curve"])
            curves["rs_mean"].append(rs_details["mean_curve"])

            print(
                "        individual_count (GA/RS): "
                f"{int(ga_metric['individual_count'])}/{int(rs_metric['individual_count'])}"
            )

        ds_dir = results_root / dataset_key
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            ds_dir / "trial_metrics.npz",
            ga_individual_count=np.asarray(metrics["individual_count"]["ga"], dtype=float),
            rs_individual_count=np.asarray(metrics["individual_count"]["rs"], dtype=float),
            ga_individual_mean=np.asarray(metrics["individual_mean"]["ga"], dtype=float),
            rs_individual_mean=np.asarray(metrics["individual_mean"]["rs"], dtype=float),
            ga_demographic_parity=np.asarray(metrics["demographic_parity"]["ga"], dtype=float),
            rs_demographic_parity=np.asarray(metrics["demographic_parity"]["rs"], dtype=float),
            ga_equalized_odds=np.asarray(metrics["equalized_odds"]["ga"], dtype=float),
            rs_equalized_odds=np.asarray(metrics["equalized_odds"]["rs"], dtype=float),
        )
        np.savez(
            ds_dir / "convergence.npz",
            ga_best=np.asarray(_pad_curves(curves["ga_best"]), dtype=float),
            ga_mean=np.asarray(_pad_curves(curves["ga_mean"]), dtype=float),
            rs_best=np.asarray(_pad_curves(curves["rs_best"]), dtype=float),
            rs_mean=np.asarray(_pad_curves(curves["rs_mean"]), dtype=float),
        )

        sensitivity = []
        if args.with_sensitivity:
            print("[5/5] Hyperparameter sensitivity analysis ...")
            sensitivity = _run_sensitivity(
                model=model,
                sampler=sampler,
                sensitive_index=sensitive_index,
                feature_ranges=feature_ranges,
                budget=budget,
                trials=args.sensitivity_trials,
                pop_sizes=pop_sizes,
                mutation_rates=mutation_rates,
                categorical_indices=categorical_indices,
            )
            (ds_dir / "sensitivity.json").write_text(
                json.dumps(sensitivity, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        dataset_entry = {
            "dataset_path": str(ds_path),
            "sensitive_attr": ds_cfg["sensitive_attr"],
            "model_accuracy": {"train": train_acc, "test": test_acc},
            "real_test_fairness": real_test_fairness,
            "metrics": metrics,
            "sensitivity": sensitivity,
        }
        combined_path = results_root / "combined_results.json"
        config_entry = {
            "trials": n_trials,
            "budget": budget,
            "ga_population": args.ga_population,
            "ga_mutation": args.ga_mutation,
        }
        _merge_into_combined(combined_path, dataset_key, dataset_entry, config_entry)
        print(f"\nDone ({dataset_key}): {combined_path}")


if __name__ == "__main__":
    main()
