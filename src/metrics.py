import numpy as np

def search_efficiency(scores, threshold):
    return np.sum(np.array(scores) > threshold)


def demographic_parity_difference(model, samples, sensitive_index, threshold=0.5):
    if samples is None or len(samples) == 0:
        return 0.0
    samples = np.asarray(samples)
    y_hat = model.predict_proba(samples)[:, 1] >= threshold
    groups = (samples[:, sensitive_index] >= 0.5).astype(int)
    if np.sum(groups == 0) == 0 or np.sum(groups == 1) == 0:
        return 0.0
    rate0 = np.mean(y_hat[groups == 0])
    rate1 = np.mean(y_hat[groups == 1])
    return float(abs(rate0 - rate1))

def equalized_odds_difference_proxy(model, samples, sensitive_index, threshold=0.5):
    """Proxy EO over generated unlabeled inputs using counterfactual predictions."""
    if samples is None or len(samples) == 0:
        return 0.0
    samples = np.asarray(samples)
    groups = (samples[:, sensitive_index] >= 0.5).astype(int)
    if np.sum(groups == 0) == 0 or np.sum(groups == 1) == 0:
        return 0.0

    y_pred = (model.predict_proba(samples)[:, 1] >= threshold).astype(int)
    flipped = samples.copy()
    flipped[:, sensitive_index] = 1 - flipped[:, sensitive_index]
    y_ref = (model.predict_proba(flipped)[:, 1] >= threshold).astype(int)

    diffs = []
    for true_label in (0, 1):
        idx = y_ref == true_label
        if np.sum(idx & (groups == 0)) == 0 or np.sum(idx & (groups == 1)) == 0:
            continue
        rate0 = np.mean(y_pred[idx & (groups == 0)])
        rate1 = np.mean(y_pred[idx & (groups == 1)])
        diffs.append(abs(rate0 - rate1))

    return float(max(diffs) if diffs else 0.0)