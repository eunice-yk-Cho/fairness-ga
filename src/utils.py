import numpy as np

def compute_feature_ranges(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return list(zip(mins.tolist(), maxs.tolist()))

def make_sampler(feature_ranges, categorical_indices=frozenset()):
    """Returns a function that generates random samples within feature ranges."""
    def sampler():
        x = np.empty(len(feature_ranges))
        for i, (lo, hi) in enumerate(feature_ranges):
            if i in categorical_indices:
                x[i] = np.random.randint(int(lo), int(hi) + 1)
            else:
                x[i] = np.random.uniform(lo, hi)
        return x
    return sampler