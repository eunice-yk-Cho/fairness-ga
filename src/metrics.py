import numpy as np

def search_efficiency(scores, threshold):
    return np.sum(np.array(scores) > threshold)

def success_rate(scores, threshold):
    scores = np.array(scores)
    return np.mean(scores > threshold)

def diversity(population):
    from scipy.spatial.distance import pdist
    return np.mean(pdist(population))