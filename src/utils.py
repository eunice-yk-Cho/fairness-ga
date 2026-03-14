import numpy as np

def compute_feature_ranges(X):
    ranges = []
    for i in range(X.shape[1]):
        ranges.append((np.min(X[:, i]), np.max(X[:, i])))
    return ranges

def make_sampler(feature_ranges):
    """Return a callable that samples a random individual from the given feature ranges."""
    def sampler():
        return np.array([np.random.uniform(lo, hi) for lo, hi in feature_ranges])
    return sampler