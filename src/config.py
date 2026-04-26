import numpy as np
import random

RANDOM_SEED = 42
POPULATION_SIZE = 50
GENERATIONS = 1000
N_TRIALS = 30
TOURNAMENT_K = 3
MUTATION_RATE = 0.1
DISCRIMINATION_THRESHOLD = 0.2
DEFAULT_DATASETS = ["adult", "compas", "german"]

DATASET_CONFIGS = {
    "adult": {
        "path": "data/adult.csv",
        "target": "income",
        "positive_label": ">50K",
        "sensitive_attr": "sex",
    },
    "compas": {
        "path": "data/compas.csv",
        "target": "two_year_recid",
        "positive_label": 1,
        "sensitive_attr": "race",
    },
    "german": {
        "path": "data/german_credit.csv",
        "target": "class",
        "positive_label": "good",
        "sensitive_attr": "sex",
    },
}

SENSITIVITY_POPULATION_SIZES = [10, 20, 40]
SENSITIVITY_MUTATION_RATES = [0.05, 0.1, 0.2]

def evaluation_budget():
    return POPULATION_SIZE * GENERATIONS

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)