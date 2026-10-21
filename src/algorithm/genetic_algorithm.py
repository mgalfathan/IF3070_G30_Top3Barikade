from __future__ import annotations

import random

from ..core.model import Problem, State
from ..core.objective import evaluate, is_valid
from ..evaluate.init import random_state
from ..evaluate.moves import random_neighbor
from dataclasses import replace
from ..evaluate.result import Result, Timer


ALGORITHM_NAME = "genetic_algorithm"


def initialize_population(problem: Problem, size: int, rng: random.Random, p_placed: float = 0.5) -> list[State]:
    """
    Membangkitkan `size` individu random. Probabilitas vehicle diletakkan dikendalikan oleh `p_placed`.
    """
    population = []

    for _ in range(size):
        population.append(random_state(problem, rng, p_placed))

    return population


def evaluate_population(problem: Problem, population: list[State]) -> list[float]:
    """
    Menghitung fitness setiap individu.
    """
    fitnesses = []

    for i in range(len(population)):
        fitnesses.append(evaluate(problem, population[i]))

    return fitnesses


def tournament_selection(population: list[State], fitnesses: list[float],
                         k: int, rng: random.Random) -> State:
    """
    Memilih satu induk dengan tournament selection.
    Peserta dengan nilai fitness negatif masih bisa terpilih.
    """
    k = min(k, len(population)) # agar tidak lebih dari ukuran populasi

    idx = rng.sample(range(len(population)), k)

    best = idx[0]

    for i in idx:
        if fitnesses[i] > fitnesses[best]:
            best = i 

    return population[best]


def crossover(parent_a: State, parent_b: State, rng: random.Random) -> State:
    """
    Menghasilkan satu anak dari dua induk.
    """
    offspring = []

    for i in range(len(parent_a.placements)):
        if rng.random() < 0.5:
            offspring.append(replace(parent_a.placements[i]))
        else:
            offspring.append(replace(parent_b.placements[i]))
 
    return State(offspring)

def mutate(problem: Problem, state: State, mutation_rate: float,
           rng: random.Random) -> State:
    """
    Memutasi individu sekali dengan peluang mutation_rate.
    """
    if rng.random() < mutation_rate:
        return random_neighbor(problem, state, rng)

    return state


def select_elites(population: list[State], fitnesses: list[float],
                  count: int) -> list[State]:
    """
    Mengambil `count` individu terbaik untuk langsung dibawa ke generasi berikutnya.
    """
    count = min(count, len(population))
    chosen = [] # daftar yang sudah terpilih
    elites = [] # copy dari individu elite

    for _ in range(count):
        best = None

        for i in range(len(population)):
            if i in chosen:
                continue
            if best is None or fitnesses[i] > fitnesses[best]:
                best = i

        chosen.append(best)
        elites.append(population[best].copy())

    return elites


def genetic_algorithm(problem: Problem,
                      population_size: int = 50,
                      iterations: int = 300,
                      crossover_rate: float = 0.8,
                      mutation_rate: float = 0.2,
                      elite_count: int = 2,
                      tournament_size: int = 3,
                      p_placed: float = 0.5,
                      seed: int | None = None) -> Result:
    """
    Menjalankan Genetic Algorithm dan mengembalikan Result.
    """
    rng = random.Random(seed)
    history: dict[str, list[float]] = {"max": [], "avg": []}
 
    with Timer() as timer:
            population = initialize_population(problem, population_size, rng, p_placed)
            fitness = evaluate_population(problem, population)
            initial_state = select_elites(population, fitness, 1)[0]
            initial_value = max(fitness)
            state_bsf = initial_state.copy()
            value_bsf = initial_value
            best_gen = 0

            for generation in range(1, iterations + 1):
                new_population = select_elites(population, fitness, elite_count)

                while len(new_population) < population_size:
                    parent_a = tournament_selection(population, fitness, tournament_size, rng)
                    parent_b = tournament_selection(population, fitness, tournament_size, rng)

                    if rng.random() < crossover_rate:
                        child = crossover(parent_a, parent_b, rng)
                    else:
                        child = parent_a.copy()

                    child = mutate(problem, child, mutation_rate, rng)
                    new_population.append(child)

                population = new_population
                fitness = evaluate_population(problem, population)

                history["max"].append(max(fitness))
                history["avg"].append(sum(fitness) / len(fitness))
                
                if max(fitness) > value_bsf:
                    state_bsf = select_elites(population, fitness, 1)[0]
                    value_bsf = max(fitness)
                    best_gen = generation
 
            final_valid = is_valid(problem, state_bsf)

    return Result(
        algorithm=ALGORITHM_NAME,
        initial_state=initial_state,
        final_state=state_bsf,
        initial_value=initial_value,
        final_value=value_bsf,
        history=history,
        duration=timer.elapsed,
        params={
            "population_size": population_size,
            "iterations": iterations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elite_count": elite_count,
            "tournament_size": tournament_size,
            "p_placed": p_placed,
            "seed": seed,
        },
        extra={
            "final_valid": final_valid,
            "best_generation": best_gen,
        },
    )