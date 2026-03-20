import numpy as np
from src.fairness import discrimination_score

def random_search(model, sampler, sensitive_index, n_iter, chunk_size=None, return_details=False):
    results = []
    samples = []
    best_curve = []
    mean_curve = []
    running_best = -np.inf
    report_every = max(1, n_iter // 10)
    chunk_size = max(1, int(chunk_size or n_iter))
    for i in range(n_iter):
        x = sampler()
        score = discrimination_score(model, x, sensitive_index)
        samples.append(x.copy())
        results.append(score)
        if (i + 1) % chunk_size == 0 or (i + 1) == n_iter:
            chunk = results[max(0, i + 1 - chunk_size):i + 1]
            running_best = max(running_best, float(np.max(chunk)))
            best_curve.append(running_best)
            mean_curve.append(float(np.mean(chunk)))
        if (i + 1) % report_every == 0 or (i + 1) == n_iter:
            print(f"      Random Search {i + 1}/{n_iter}")
    if return_details:
        return {
            "scores": results,
            "samples": np.array(samples),
            "best_curve": best_curve,
            "mean_curve": mean_curve,
        }
    return results