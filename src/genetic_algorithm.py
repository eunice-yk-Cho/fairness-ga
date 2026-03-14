import numpy as np
import random
from src.fairness import discrimination_score

class GeneticAlgorithm:

    def __init__(self, model, sampler, sensitive_index, config, feature_ranges=None):
        self.model = model
        self.sampler = sampler
        self.sensitive_index = sensitive_index
        self.config = config
        self.feature_ranges = feature_ranges

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
        if self.feature_ranges is not None:
            lo = np.array([r[0] for r in self.feature_ranges])
            hi = np.array([r[1] for r in self.feature_ranges])
            child = np.clip(child, lo, hi)
        return child

    def mutate(self, x):
        if np.random.rand() < self.config.MUTATION_RATE:
            idx = np.random.randint(len(x))
            x[idx] += np.random.normal(0, 0.1)
            if self.feature_ranges is not None:
                lo, hi = self.feature_ranges[idx]
                x[idx] = np.clip(x[idx], lo, hi)
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
            "Cannot infer evaluation budget. Provide n_evals, or set config.evaluation_budget(), "
            "or define config.POPULATION_SIZE and config.GENERATIONS."
        )

    def run(self, n_evals=None, return_details=False):
        population = self.initialize_population()
        scores = []
        samples = []
        best_curve = []
        mean_curve = []
        max_evals = self._resolve_max_evals(n_evals)
        report_every = max(1, max_evals // 10)  # 최대 10번 진행 출력
        next_report = report_every

        for _ in range(self.config.GENERATIONS):
            new_population = []
            generation_scores = []
            for _ in range(len(population)):
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
                score = self.fitness(child)
                scores.append(score)
                samples.append(child.copy())
                generation_scores.append(score)
                n_done = len(scores)

                if n_done >= next_report or n_done == max_evals:
                    print(f"      GA 평가 {n_done}/{max_evals}")
                    while next_report <= n_done:
                        next_report += report_every

                if n_done >= max_evals:
                    if generation_scores:
                        best_curve.append(float(np.max(generation_scores)))
                        mean_curve.append(float(np.mean(generation_scores)))
                    if return_details:
                        return {
                            "scores": scores,
                            "samples": np.array(samples),
                            "best_curve": best_curve,
                            "mean_curve": mean_curve,
                        }
                    return scores

            population = new_population
            if generation_scores:
                best_curve.append(float(np.max(generation_scores)))
                mean_curve.append(float(np.mean(generation_scores)))

        if return_details:
            return {
                "scores": scores,
                "samples": np.array(samples),
                "best_curve": best_curve,
                "mean_curve": mean_curve,
            }
        return scores