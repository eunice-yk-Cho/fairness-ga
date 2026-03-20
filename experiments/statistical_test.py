# -*- coding: utf-8 -*-
import json
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_out_txt = _root / "experiments" / "stat_test_summary.txt"
_combined_path = _root / "experiments" / "results" / "combined_results.json"

def safe_wilcoxon_greater(ga, rs):
    ga = np.asarray(ga, dtype=float)
    rs = np.asarray(rs, dtype=float)
    if len(ga) != len(rs) or len(ga) == 0:
        raise ValueError("GA and Random arrays must have the same length.")
    diff = ga - rs
    if np.allclose(diff, 0.0):
        return 0.0, 1.0, "all_zero_diff"
    stat, p = wilcoxon(ga, rs, alternative="greater")
    return float(stat), float(p), "ok"

def main():
    if not _combined_path.exists():
        print(f"Missing file: {_combined_path}")
        print("Run `python experiments/run_experiment.py` first.")
        raise SystemExit(1)

    combined = json.loads(_combined_path.read_text(encoding="utf-8"))
    metric_keys = ["individual_count", "individual_mean", "demographic_parity", "equalized_odds"]

    lines = []
    for dataset_key, info in combined.get("datasets", {}).items():
        lines.append(f"[Dataset] {dataset_key}")
        print(f"[Dataset] {dataset_key}")
        metrics = info.get("metrics", {})

        for metric_key in metric_keys:
            ga = np.asarray(metrics.get(metric_key, {}).get("ga", []), dtype=float)
            rs = np.asarray(metrics.get(metric_key, {}).get("rs", []), dtype=float)
            if len(ga) == 0 or len(rs) == 0:
                continue

            stat, p, mode = safe_wilcoxon_greater(ga, rs)
            line_block = [
                f"  [Metric] {metric_key}",
                f"    N trials: {len(ga)}",
                f"    GA mean±std: {ga.mean():.4f} ± {ga.std():.4f}",
                f"    RS mean±std: {rs.mean():.4f} ± {rs.std():.4f}",
                f"    Mean difference (GA-RS): {(ga - rs).mean():.4f}",
                f"    Win rate (GA>RS): {np.mean(ga > rs):.3f}",
                f"    Wilcoxon statistic: {stat}",
                f"    p-value: {p}",
            ]
            if mode == "all_zero_diff":
                line_block.append("    Note: All paired differences are zero; p=1.0.")
            line_block.append("")

            lines.extend(line_block)
            for line in line_block:
                print(line)

        lines.append("")
        print()

    _out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {_out_txt}")


if __name__ == "__main__":
    main()