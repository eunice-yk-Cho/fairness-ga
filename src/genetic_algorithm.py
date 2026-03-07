import numpy as np
import random
from src.fairness import discrimination_score

class GeneticAlgorithm:

    def __init__(self, model, sampler, sensitive_index, config):
        self.model = model
        self.sampler = sampler
        self.sensitive_index = sensitive_index
        self.config = config

    def initialize_population(self):
        return [self.sampler() for _ in range(self.config.POPULATION_SIZE)]

    def fitness(self, x):
        return discrimination_score(self.model, x, self.sensitive_index)

    def tournament_selection(self, population):
        selected = random.sample(population, self.config.TOURNAMENT_K)
        selected.sort(key=lambda x: self.fitness(x), reverse=True)
        return selected[0]

    def crossover(self, p1, p2):
        mask = np.random.rand(len(p1)) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def mutate(self, x):
        if np.random.rand() < self.config.MUTATION_RATE:
            idx = np.random.randint(len(x))
            x[idx] += np.random.normal(0, 0.1)
        return x

    def run(self):
        population = self.initialize_population()
        scores = []
        n_gen = self.config.GENERATIONS
        report_every = max(1, n_gen // 10)  # 최대 10번 진행 출력

        for g in range(n_gen):
            new_population = []
            for _ in range(len(population)):
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
                scores.append(self.fitness(child))
            population = new_population

            if (g + 1) % report_every == 0 or (g + 1) == n_gen:
                print(f"      GA 세대 {g + 1}/{n_gen} (평가 {len(scores)}회)")

        return scores