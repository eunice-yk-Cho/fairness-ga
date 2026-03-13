import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

ga_counts_path = _root / "ga_trial_counts.npy"
rs_counts_path = _root / "random_trial_counts.npy"
ga_means_path = _root / "ga_trial_means.npy"
rs_means_path = _root / "random_trial_means.npy"

if ga_counts_path.exists() and rs_counts_path.exists():
    ga_counts = np.load(ga_counts_path)
    rs_counts = np.load(rs_counts_path)
    stat_c, p_c = wilcoxon(ga_counts, rs_counts, alternative="greater")

    print("[Metric] Discriminatory cases per trial")
    print("GA mean±std:", f"{ga_counts.mean():.2f} ± {ga_counts.std():.2f}")
    print("RS mean±std:", f"{rs_counts.mean():.2f} ± {rs_counts.std():.2f}")
    print("Wilcoxon statistic:", stat_c)
    print("p-value:", p_c)
    print()

if ga_means_path.exists() and rs_means_path.exists():
    ga_means = np.load(ga_means_path)
    rs_means = np.load(rs_means_path)
    stat_m, p_m = wilcoxon(ga_means, rs_means, alternative="greater")

    print("[Metric] Mean discrimination score per trial")
    print("GA mean±std:", f"{ga_means.mean():.4f} ± {ga_means.std():.4f}")
    print("RS mean±std:", f"{rs_means.mean():.4f} ± {rs_means.std():.4f}")
    print("Wilcoxon statistic:", stat_m)
    print("p-value:", p_m)
    print()

if not (ga_counts_path.exists() and rs_counts_path.exists()) and not (ga_means_path.exists() and rs_means_path.exists()):
    ga_results = np.load(_root / "ga_results.npy")
    random_results = np.load(_root / "random_results.npy")
    if len(ga_results) != len(random_results):
        print("[Fallback] Raw score comparison skipped")
        print("ga_results and random_results have different lengths.")
        print("Run `python experiments/run_experiment.py` to regenerate matched trial outputs.")
        raise SystemExit(1)
    stat, p = wilcoxon(ga_results, random_results, alternative="greater")
    print("[Fallback] Raw score comparison")
    print("Wilcoxon statistic:", stat)
    print("p-value:", p)