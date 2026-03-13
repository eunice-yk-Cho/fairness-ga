import numpy as np
import random

RANDOM_SEED = 42
POPULATION_SIZE = 50
GENERATIONS = 1000
N_TRIALS = 30
TOURNAMENT_K = 3
MUTATION_RATE = 0.1
DISCRIMINATION_THRESHOLD = 0.2
SENSITIVE_ATTR = "sex"

def evaluation_budget():
    return POPULATION_SIZE * GENERATIONS

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)