import numpy as np
from src.fairness import discrimination_score

def random_search(model, sampler, sensitive_index, n_iter):
    results = []
    report_every = max(1, n_iter // 10)  # 최대 10번 진행 출력
    for i in range(n_iter):
        x = sampler()
        score = discrimination_score(model, x, sensitive_index)
        results.append(score)
        if (i + 1) % report_every == 0 or (i + 1) == n_iter:
            print(f"      Random Search {i + 1}/{n_iter}")
    return results