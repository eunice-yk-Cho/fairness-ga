import numpy as np
import random
from src.fairness import discrimination_score


class GeneticAlgorithm:

    def __init__(self, model, sampler, sensitive_index, config,
                 feature_ranges=None, categorical_indices=frozenset()):
        pop = getattr(config, "POPULATION_SIZE", None)
        tk = getattr(config, "TOURNAMENT_K", None)
        mr = getattr(config, "MUTATION_RATE", None)
        if pop is not None and pop < 1:
            raise ValueError(f"POPULATION_SIZE must be >= 1, got {pop}")
        if tk is not None and pop is not None and tk > pop:
            raise ValueError(
                f"TOURNAMENT_K ({tk}) must be <= POPULATION_SIZE ({pop})")
        if mr is not None and not (0.0 <= mr <= 1.0):
            raise ValueError(f"MUTATION_RATE must be in [0, 1], got {mr}")
        self.model = model
        self.sampler = sampler
        self.sensitive_index = sensitive_index
        self.config = config
        self.feature_ranges = feature_ranges
        self.categorical_indices = categorical_indices

    def initialize_population(self):
        return [self.sampler() for _ in range(self.config.POPULATION_SIZE)]

    def fitness(self, x):
        return discrimination_score(self.model, x, self.sensitive_index)

    def tournament_selection(self, population, fitness_values):
        """Pick TOURNAMENT_K individuals at random and return the fittest."""
        indices = random.sample(range(len(population)), self.config.TOURNAMENT_K)
        best_idx = max(indices, key=lambda i: fitness_values[i])
        return population[best_idx]

    def crossover(self, p1, p2):
        mask = np.random.rand(len(p1)) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def mutate(self, x):
        x = x.copy()
        if np.random.rand() < self.config.MUTATION_RATE:
            idx = np.random.randint(len(x))
            if idx in self.categorical_indices:
                lo, hi = self.feature_ranges[idx]
                x[idx] = np.random.randint(int(lo), int(hi) + 1)
            elif self.feature_ranges is not None:
                lo, hi = self.feature_ranges[idx]
                sigma = (hi - lo) * 0.05
                x[idx] += np.random.normal(0, max(sigma, 1e-9))
                x[idx] = np.clip(x[idx], lo, hi)
            else:
                x[idx] += np.random.normal(0, 0.1)
        return x

    def _resolve_max_evals(self, n_evals):
        if n_evals is not None:
            return int(n_evals)

        budget_fn = getattr(self.config, "evaluation_budget", None)
        if callable(budget_fn):
            return int(budget_fn())

        pop_size = getattr(self.config, "POPULATION_SIZE", None)
        generations = getattr(self.config, "GENERATIONS", None)
        if pop_size is not None and generations is not None:
            return int(pop_size) * int(generations)

        raise AttributeError(
            "Cannot infer evaluation budget. Provide n_evals, or set "
            "config.evaluation_budget(), or define config.POPULATION_SIZE "
            "and config.GENERATIONS."
        )

    def run(self, n_evals=None, return_details=False):
        max_evals = self._resolve_max_evals(n_evals)
        if max_evals <= 0:
            raise ValueError(f"max_evals must be > 0, got {max_evals}")
        report_every = max(1, max_evals // 10)
        next_report = report_every

        init_size = min(self.config.POPULATION_SIZE, max_evals)
        population = [self.sampler() for _ in range(init_size)]
        pop_fitness = [self.fitness(x) for x in population]

        all_scores = list(pop_fitness)
        all_samples = [x.copy() for x in population]
        running_best = float(np.max(pop_fitness))
        best_curve = [running_best]
        mean_curve = [float(np.mean(pop_fitness))]
        eval_count = len(population)

        if eval_count >= next_report:
            print(f"      GA eval {eval_count}/{max_evals}")
            while next_report <= eval_count:
                next_report += report_every

        while eval_count < max_evals:
            elite_idx = int(np.argmax(pop_fitness))
            elite = population[elite_idx].copy()
            elite_fit = pop_fitness[elite_idx]

            new_population = [elite]
            new_fitness = [elite_fit]
            gen_scores = []

            for _ in range(self.config.POPULATION_SIZE - 1):
                if eval_count >= max_evals:
                    break

                p1 = self.tournament_selection(population, pop_fitness)
                p2 = self.tournament_selection(population, pop_fitness)
                child = self.crossover(p1, p2)
                child = self.mutate(child)

                score = self.fitness(child)
                eval_count += 1

                new_population.append(child)
                new_fitness.append(score)
                all_scores.append(score)
                all_samples.append(child.copy())
                gen_scores.append(score)

                if eval_count >= next_report or eval_count == max_evals:
                    print(f"      GA eval {eval_count}/{max_evals}")
                    while next_report <= eval_count:
                        next_report += report_every

            if gen_scores:
                gen_best = float(np.max(gen_scores))
                running_best = max(running_best, gen_best)
                best_curve.append(running_best)
                mean_curve.append(float(np.mean(gen_scores)))

            population = new_population
            pop_fitness = new_fitness

        if return_details:
            return {
                "scores": all_scores,
                "samples": np.array(all_samples),
                "best_curve": best_curve,
                "mean_curve": mean_curve,
            }
        return all_scores
