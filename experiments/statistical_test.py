import numpy as np
from scipy.stats import wilcoxon

ga_results = np.load("ga_results.npy")
random_results = np.load("random_results.npy")

stat, p = wilcoxon(ga_results, random_results, alternative='greater')

print("Wilcoxon statistic:", stat)
print("p-value:", p)