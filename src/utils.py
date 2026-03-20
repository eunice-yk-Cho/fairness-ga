import numpy as np

def compute_feature_ranges(X):
    # FIX #31: vectorized min/max instead of per-column loop
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return list(zip(mins.tolist(), maxs.tolist()))

def make_sampler(feature_ranges, categorical_indices=frozenset()):
    """Return a callable that samples a random individual from the given feature ranges.

    Categorical features (identified by *categorical_indices*) are sampled as
    random integers within [lo, hi] so that generated inputs remain on the same
    discrete support the model was trained on.
    """
    def sampler():
        x = np.empty(len(feature_ranges))
        for i, (lo, hi) in enumerate(feature_ranges):
            if i in categorical_indices:
                x[i] = np.random.randint(int(lo), int(hi) + 1)
            else:
                x[i] = np.random.uniform(lo, hi)
        return x
    return sampler