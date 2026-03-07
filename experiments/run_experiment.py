import sys
from pathlib import Path

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

print("[3/6] GA 실행 중 (세대마다 진행 출력)...")
ga = GeneticAlgorithm(model, sampler, sensitive_index, config)
ga_scores = ga.run()
print(f"      GA 완료: {len(ga_scores)}개 점수 수집")

print("[4/6] Random Search 실행 중...")
rs_scores = random_search(model, sampler, sensitive_index, config.GENERATIONS)
print(f"      Random Search 완료: {len(rs_scores)}개 점수 수집")

print("[5/6] 결과 저장 중...")
np.save(_root / "ga_results.npy", np.array(ga_scores))
np.save(_root / "random_results.npy", np.array(rs_scores))
print(f"      저장 경로: {_root}")

print("[6/6] 요약")
print("GA discriminatory cases:",
      search_efficiency(ga_scores, config.DISCRIMINATION_THRESHOLD))

print("Random discriminatory cases:",
      search_efficiency(rs_scores, config.DISCRIMINATION_THRESHOLD))