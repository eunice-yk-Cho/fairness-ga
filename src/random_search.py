import numpy as np

def random_search(model, sampler, sensitive_index, n_iter, chunk_size=None, return_details=False):
    best_curve = []
    mean_curve = []
    running_best = -np.inf
    chunk_size = max(1, int(chunk_size or n_iter))

    # Generate all samples upfront for batch evaluation
    samples_list = [sampler() for _ in range(n_iter)]
    samples = np.array(samples_list)
    samples_flip = samples.copy()
    samples_flip[:, sensitive_index] = 1 - samples_flip[:, sensitive_index]
    p1 = model.predict_proba(samples)[:, 1]
    p2 = model.predict_proba(samples_flip)[:, 1]
    scores = list(np.abs(p1 - p2))

    print(f"      Random Search {n_iter}/{n_iter}")

    # Build convergence curves in chunks
    n_full_chunks = n_iter // chunk_size
    for c in range(n_full_chunks):
        start = c * chunk_size
        chunk = scores[start:start + chunk_size]
        running_best = max(running_best, float(np.max(chunk)))
        best_curve.append(running_best)
        mean_curve.append(float(np.mean(chunk)))

    remainder_start = n_full_chunks * chunk_size
    if remainder_start < n_iter:
        chunk = scores[remainder_start:]
        running_best = max(running_best, float(np.max(chunk)))
        best_curve.append(running_best)
        mean_curve.append(float(np.mean(chunk)))

    if return_details:
        return {
            "scores": scores,
            "samples": samples,
            "best_curve": best_curve,
            "mean_curve": mean_curve,
        }
    return scores
