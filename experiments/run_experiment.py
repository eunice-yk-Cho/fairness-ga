import sys
from pathlib import Path
import argparse

# 프로젝트 루트를 경로에 추가 (python experiments/run_experiment.py 실행 시 src 인식)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from src import config
from src.preprocessing import load_and_preprocess
from src.model import train_model
from src.utils import compute_feature_ranges, make_sampler
from src.random_search import random_search
from src.genetic_algorithm import GeneticAlgorithm
from src.metrics import search_efficiency

parser = argparse.ArgumentParser(description="Run GA vs Random fairness experiment with matched budgets.")
parser.add_argument("--trials", type=int, default=config.N_TRIALS, help="Number of independent trials.")
parser.add_argument("--budget", type=int, default=config.evaluation_budget(), help="Evaluations per method per trial.")
args = parser.parse_args()

config.set_seed()

print("[1/6] 데이터 로드 중...")
X_train, X_test, y_train, y_test, sensitive_index = \
    load_and_preprocess("data/adult.csv", config.SENSITIVE_ATTR)
print(f"      train {X_train.shape[0]}건, test {X_test.shape[0]}건, sensitive_index={sensitive_index}")

print("[2/6] 모델 학습 중...")
model = train_model(X_train, y_train)
print("      완료")

feature_ranges = compute_feature_ranges(X_train)
sampler = make_sampler(feature_ranges)

budget = args.budget
n_trials = args.trials

print(f"[3/6] 반복 실험 실행 중... (trial={n_trials}, budget={budget}/trial)")
ga_trial_counts = []
rs_trial_counts = []
ga_trial_means = []
rs_trial_means = []
all_ga_scores = []
all_rs_scores = []

for t in range(n_trials):
    trial_seed = config.RANDOM_SEED + t
    config.set_seed(trial_seed)
    print(f"      Trial {t + 1}/{n_trials} (seed={trial_seed})")

    ga = GeneticAlgorithm(model, sampler, sensitive_index, config)
    ga_scores = ga.run()
    rs_scores = random_search(model, sampler, sensitive_index, budget)

    ga_count = int(search_efficiency(ga_scores, config.DISCRIMINATION_THRESHOLD))
    rs_count = int(search_efficiency(rs_scores, config.DISCRIMINATION_THRESHOLD))

    ga_trial_counts.append(ga_count)
    rs_trial_counts.append(rs_count)
    ga_trial_means.append(float(np.mean(ga_scores)))
    rs_trial_means.append(float(np.mean(rs_scores)))
    all_ga_scores.extend(ga_scores)
    all_rs_scores.extend(rs_scores)

    print(f"        discriminatory cases (GA/RS): {ga_count}/{rs_count}")

print("[5/6] 결과 저장 중...")
np.save(_root / "ga_results.npy", np.array(all_ga_scores))
np.save(_root / "random_results.npy", np.array(all_rs_scores))
np.save(_root / "ga_trial_counts.npy", np.array(ga_trial_counts))
np.save(_root / "random_trial_counts.npy", np.array(rs_trial_counts))
np.save(_root / "ga_trial_means.npy", np.array(ga_trial_means))
np.save(_root / "random_trial_means.npy", np.array(rs_trial_means))
print(f"      저장 경로: {_root}")

print("[6/6] 요약")
print(f"GA discriminatory cases (mean±std): {np.mean(ga_trial_counts):.2f} ± {np.std(ga_trial_counts):.2f}")
print(f"Random discriminatory cases (mean±std): {np.mean(rs_trial_counts):.2f} ± {np.std(rs_trial_counts):.2f}")
print(f"GA mean discrimination score (mean±std): {np.mean(ga_trial_means):.4f} ± {np.std(ga_trial_means):.4f}")
print(f"Random mean discrimination score (mean±std): {np.mean(rs_trial_means):.4f} ± {np.std(rs_trial_means):.4f}")