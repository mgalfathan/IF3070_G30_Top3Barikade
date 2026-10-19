from __future__ import annotations
import math
import random
from ..core.model import Problem, State
from ..core.objective import evaluate
from ..evaluate.init import random_state
from ..evaluate.moves import random_neighbor
from ..evaluate.result import Result, Timer


def default_t0(problem: Problem) -> float:
    return max(1.0, sum(v.fee for v in problem.vehicles) / max(1, problem.n))


def simulated_annealing(problem: Problem,
                        initial: State | None = None,
                        t0: float | None = None,
                        alpha: float = 0.995,
                        t_min: float = 1e-3,
                        max_iter: int = 20_000,
                        stuck_patience: int = 100,
                        p_placed: float = 0.5,
                        seed: int | None = None) -> Result:
    rng = random.Random(seed)
    t0 = default_t0(problem) if t0 is None else t0

    current = initial.copy() if initial is not None else random_state(problem, rng, p_placed)
    current_value = evaluate(problem, current)
    initial_state, initial_value = current.copy(), current_value
    best, best_value = current.copy(), current_value

    history = {"value": [], "best": [], "temperature": [], "accept_prob": []}
    accepted_worse = 0
    rejected = 0
    stuck_count = 0
    stuck_iterations = 0
    streak = 0

    t = t0
    with Timer() as timer:
        for _ in range(max_iter):
            if t < t_min:
                break

            neighbor = random_neighbor(problem, current, rng)
            neighbor_value = evaluate(problem, neighbor)
            delta = neighbor_value - current_value
            prob = 1.0 if delta >= 0 else math.exp(delta / t)
            previous_value = current_value

            if delta >= 0 or rng.random() < prob:
                if delta < 0:
                    accepted_worse += 1
                current, current_value = neighbor, neighbor_value
            else:
                rejected += 1

            if current_value > best_value:
                best, best_value = current.copy(), current_value

            # stuck: nilai current tidak berubah selama stuck_patience iterasi
            if current_value != previous_value:
                streak = 0
            else:
                streak += 1
                if streak >= stuck_patience:
                    stuck_iterations += 1
                    if streak == stuck_patience:
                        stuck_count += 1

            history["value"].append(current_value)
            history["best"].append(best_value)
            history["temperature"].append(t)
            history["accept_prob"].append(prob)

            t *= alpha

    return Result(
        algorithm="simulated_annealing",
        initial_state=initial_state,
        final_state=best,
        initial_value=initial_value,
        final_value=best_value,
        history=history,
        duration=timer.elapsed,
        params={"t0": t0, "alpha": alpha, "t_min": t_min, "max_iter": max_iter,
                "stuck_patience": stuck_patience, "p_placed": p_placed, "seed": seed},
        extra={
            "stuck_count": stuck_count,
            "stuck_iterations": stuck_iterations,
            "accepted_worse": accepted_worse,
            "rejected": rejected,
            "last_state_value": current_value,
        },
    )
