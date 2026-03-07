import numpy as np
import random

RANDOM_SEED = 42
POPULATION_SIZE = 50
GENERATIONS = 1000
TOURNAMENT_K = 3
MUTATION_RATE = 0.1
DISCRIMINATION_THRESHOLD = 0.2
SENSITIVE_ATTR = "sex"

def set_seed():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)