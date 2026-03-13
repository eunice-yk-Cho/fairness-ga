"""
GA vs Random Search 차별 점수 분포 시각화.
프로젝트 루트에서 실행: python experiments/visualization.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import matplotlib.pyplot as plt
from src import config

# 결과 로드 (루트 기준)
ga_scores = np.load(_root / "ga_results.npy")
rs_path = _root / "random_results.npy"
if rs_path.exists():
    rs_scores = np.load(rs_path)
    has_rs = True
else:
    rs_scores = np.array([])
    has_rs = False
threshold = config.DISCRIMINATION_THRESHOLD
ga_trial_counts_path = _root / "ga_trial_counts.npy"
rs_trial_counts_path = _root / "random_trial_counts.npy"
has_trial_counts = ga_trial_counts_path.exists() and rs_trial_counts_path.exists()
if has_trial_counts:
    ga_trial_counts = np.load(ga_trial_counts_path)
    rs_trial_counts = np.load(rs_trial_counts_path)

# 한글 폰트 설정 (시스템에 없으면 기본 폰트 사용)
plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1) 히스토그램: GA vs Random, 기준선
ax = axes[0, 0]
ax.hist(ga_scores, bins=50, alpha=0.6, label=f"GA (n={len(ga_scores):,})", color="C0", density=True)
if has_rs and len(rs_scores) > 0:
    ax.hist(rs_scores, bins=50, alpha=0.6, label=f"Random (n={len(rs_scores):,})", color="C1", density=True)
ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"기준 (>{threshold})")
ax.set_xlabel("차별 점수 (Discrimination Score)")
ax.set_ylabel("밀도")
ax.set_title("차별 점수 분포")
ax.legend()
ax.grid(True, alpha=0.3)

# 2) 박스플롯: GA vs Random
ax = axes[0, 1]
data_for_box = [ga_scores]
labels_box = ["GA"]
if has_rs and len(rs_scores) > 0:
    data_for_box.append(rs_scores)
    labels_box.append("Random Search")
bp = ax.boxplot(data_for_box, labels=labels_box, patch_artist=True, showfliers=False)
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(["C0", "C1"][i % 2])
ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"기준 {threshold}")
ax.set_ylabel("차별 점수")
ax.set_title("점수 분포 비교 (박스플롯)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 3) 기준 초과 개수 (Search Efficiency)
ax = axes[1, 0]
ga_count = int(np.sum(ga_scores > threshold))
rs_count = int(np.sum(rs_scores > threshold)) if has_rs else 0
bar_labels = ["GA", "Random Search"] if has_rs else ["GA"]
bar_values = [ga_count, rs_count] if has_rs else [ga_count]
bar_colors = ["C0", "C1"] if has_rs else ["C0"]
bars = ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor="black", linewidth=1.2)
ax.set_ylabel("차별 사례 수 (score > {})".format(threshold))
ax.set_title("Search Efficiency (불공정 사례 탐지 개수)")
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(int(b.get_height())), ha="center", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")

# 4) 요약 통계 텍스트
ax = axes[1, 1]
if has_trial_counts:
    bp = ax.boxplot([ga_trial_counts, rs_trial_counts], tick_labels=["GA", "Random"], patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("C0")
    bp["boxes"][1].set_facecolor("C1")
    ax.set_ylabel("불공정 사례 수 / trial")
    ax.set_title("반복 실험 분포 (trial-level)")
    ax.grid(True, alpha=0.3, axis="y")
else:
    ax.axis("off")
    summary = [
        "【요약 통계】",
        f"  GA:     min={ga_scores.min():.3f}, max={ga_scores.max():.3f}, mean={ga_scores.mean():.3f}",
    ]
    if has_rs and len(rs_scores) > 0:
        summary.append(f"  Random: min={rs_scores.min():.3f}, max={rs_scores.max():.3f}, mean={rs_scores.mean():.3f}")
    summary.extend([
        "",
        f"  기준(threshold) = {threshold}",
        f"  GA 차별 사례 수:     {ga_count}",
    ])
    if has_rs:
        summary.append(f"  Random 차별 사례 수: {rs_count}")
    ax.text(0.1, 0.9, "\n".join(summary), transform=ax.transAxes, fontsize=11, verticalalignment="top", family="monospace")

plt.suptitle("유전 알고리즘 vs 무작위 탐색: 공정성 결함 탐색 결과", fontsize=12)
plt.tight_layout()
plt.savefig(_root / "experiments" / "fairness_ga_results.png", dpi=150, bbox_inches="tight")
print(f"저장: {_root / 'experiments' / 'fairness_ga_results.png'}")
plt.show()
