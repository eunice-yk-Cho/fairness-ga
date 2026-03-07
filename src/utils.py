import numpy as np

def compute_feature_ranges(X):
    ranges = []
    for i in range(X.shape[1]):
        ranges.append((np.min(X[:, i]), np.max(X[:, i])))
    return ranges

def make_sampler(feature_ranges):
    """feature_ranges [(min, max), ...] 로부터 무작위 개체를 샘플링하는 함수 반환."""
    def sampler():
        return np.array([np.random.uniform(lo, hi) for lo, hi in feature_ranges])
    return sampler