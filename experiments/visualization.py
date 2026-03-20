# -*- coding: utf-8 -*-
"""
Visualize experiment results:
- trial-level fairness metrics (GA vs Random)
- convergence curves
- (optional) hyperparameter sensitivity heatmap

Run from project root:  python experiments/visualization.py
"""
import sys
from pathlib import Path
import argparse
import json

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import matplotlib.pyplot as plt

def _plot_convergence(ax, ga_curve, rs_curve, title, ylabel):
    ga_steps = np.arange(1, ga_curve.shape[1] + 1)
    rs_steps = np.arange(1, rs_curve.shape[1] + 1)
    ga_mu, ga_sd = ga_curve.mean(axis=0), ga_curve.std(axis=0)
    rs_mu, rs_sd = rs_curve.mean(axis=0), rs_curve.std(axis=0)
    ax.plot(ga_steps, ga_mu, color="C0", label="GA")
    ax.fill_between(ga_steps, ga_mu - ga_sd, ga_mu + ga_sd, color="C0", alpha=0.2)
    ax.plot(rs_steps, rs_mu, color="C1", label="Random")
    ax.fill_between(rs_steps, rs_mu - rs_sd, rs_mu + rs_sd, color="C1", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Chunk / generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Visualize advanced experiment results.")
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset key under experiments/results/")
    args = parser.parse_args()
    dataset_key = args.dataset

    results_dir = _root / "experiments" / "results" / dataset_key
    metrics_path = results_dir / "trial_metrics.npz"
    conv_path = results_dir / "convergence.npz"
    sensitivity_path = results_dir / "sensitivity.json"

    if not metrics_path.exists() or not conv_path.exists():
        print(f"Missing result files in: {results_dir}")
        print("Run `python experiments/run_experiment.py` first.")
        raise SystemExit(1)

    trial_metrics = np.load(metrics_path)
    conv = np.load(conv_path)

    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    metric_order = ["individual_count", "individual_mean", "demographic_parity", "equalized_odds"]
    ga_data = [trial_metrics[f"ga_{m}"] for m in metric_order]
    rs_data = [trial_metrics[f"rs_{m}"] for m in metric_order]
    x = np.arange(len(metric_order))

    ga_pos = x - 0.18
    rs_pos = x + 0.18
    for i in range(len(metric_order)):
        ax.boxplot(ga_data[i], positions=[ga_pos[i]], widths=0.3, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="C0", alpha=0.7))
        ax.boxplot(rs_data[i], positions=[rs_pos[i]], widths=0.3, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="C1", alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(["ID count", "ID mean", "DP diff", "EO diff"])
    ax.set_title("Trial-level fairness metrics")
    ax.grid(True, alpha=0.3, axis="y")
    ax.plot([], [], color="C0", linewidth=8, alpha=0.7, label="GA")
    ax.plot([], [], color="C1", linewidth=8, alpha=0.7, label="Random")
    ax.legend(loc="upper right")

    _plot_convergence(axes[0, 1], conv["ga_best"], conv["rs_best"],
                      "Convergence (best)", "Discrimination score")

    _plot_convergence(axes[1, 1], conv["ga_mean"], conv["rs_mean"],
                      "Convergence (mean per chunk)", "Discrimination score")

    ax = axes[1, 0]
    if sensitivity_path.exists():
        records = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        pop_sizes = sorted({int(r["population_size"]) for r in records})
        mutation_rates = sorted({float(r["mutation_rate"]) for r in records})
        heat = np.zeros((len(pop_sizes), len(mutation_rates)))
        for r in records:
            i = pop_sizes.index(int(r["population_size"]))
            j = mutation_rates.index(float(r["mutation_rate"]))
            heat[i, j] = float(r["mean_discriminatory_cases"])
        im = ax.imshow(heat, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(mutation_rates)))
        ax.set_xticklabels([f"{m:.2f}" for m in mutation_rates])
        ax.set_yticks(np.arange(len(pop_sizes)))
        ax.set_yticklabels([str(p) for p in pop_sizes])
        ax.set_xlabel("Mutation rate")
        ax.set_ylabel("Population size")
        ax.set_title("Sensitivity (mean ID cases)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis("off")
        ax.text(
            0.05,
            0.95,
            "Sensitivity results not found.\nRun with --with-sensitivity.",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
        )

    plt.suptitle(f"Advanced fairness results: {dataset_key}", fontsize=12)
    plt.tight_layout()
    out_path = _root / "experiments" / f"fairness_ga_results_{dataset_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
