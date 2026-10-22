from __future__ import annotations
import math
import random
from ..core.model import Problem, State
from ..core.objective import evaluate
from ..evaluate.init import random_state
from ..evaluate.moves import random_neighbor
from ..evaluate.result import Result, Timer

T_MIN = 1e-3
MAX_ITER = 20000
CHECK_EVERY = 50
SAMPLES = 30

def default_t0(problem: Problem) -> float:
    # heuristik: temperatur awal = rata-rata fee kendaraan (minimal 1)
    return max(1.0, sum(v.fee for v in problem.vehicles) / max(1, problem.n))


def is_local_optimum(problem: Problem, state: State, value: float,
                     rng: random.Random, samples: int = 30) -> bool:
    for i in range(samples):
        neighbor = random_neighbor(problem, state, rng)
        if evaluate(problem, neighbor) > value:
            return False
    return True

def simulated_annealing(problem: Problem, t0: float | None = None,
                        alpha: float = 0.995, seed: int | None = None) -> Result:

    rng = random.Random(seed)
    check_stuck_rng = random.Random(None if seed is None else seed + 1) # rng terpisah agar pengecekan stuck tidak mengubah jalannya pencarian

    if t0 is None:
        t0 = default_t0(problem)

    current_state = random_state(problem, rng)
    current_value = evaluate(problem, current_state)

    initial_state = current_state.copy()
    initial_value = current_value
    best_state = current_state.copy()
    best_value = current_value

    t = t0
    checks = 0
    stuck_count = 0
    accepted_worse = 0
    rejected = 0

    history = {"value": [], "best": [], "temperature": [], "accept_prob": []}

    with Timer() as timer:
        for i in range(1, MAX_ITER + 1):
            # 1. bangkitkan neighbor dan hitung selisih nilai
            neighbor = random_neighbor(problem, current_state, rng)
            neighbor_value = evaluate(problem, neighbor)
            delta = neighbor_value - current_value

            # 2. lebih baik: pasti diterima, lebih buruk: diterima dengan peluang e^(delta/T)
            prob = 1.0 if delta > 0 else math.exp(delta / t)
            accepted = rng.random() < prob

            if accepted:
                if delta < 0:
                    accepted_worse += 1
                current_state = neighbor
                current_value = neighbor_value
            else:
                rejected += 1

            # 3. simpan state terbaik
            if current_value > best_value:
                best_state  = current_state.copy()
                best_value = current_value

            # 4. cek stuck di local optimum setiap check_every iterasi
            if i % CHECK_EVERY == 0:
                checks += 1
                if is_local_optimum(problem, current_state, current_value, check_stuck_rng, SAMPLES):
                    stuck_count += 1

            history["value"].append(current_value)
            history["best"].append(best_value)
            history["temperature"].append(t)
            history["accept_prob"].append(prob)

            t *= alpha
            if t < T_MIN:
                break

    return Result(
        algorithm="simulated_annealing",
        initial_state=initial_state,
        final_state=best_state,
        initial_value=initial_value,
        final_value=best_value,
        history=history,
        duration=timer.elapsed,
        params={"t0": t0, "alpha": alpha, "t_min": T_MIN, "max_iter": MAX_ITER,
                "check_every": CHECK_EVERY, "samples": SAMPLES, "seed": seed},
        extra={
            "stuck_count": stuck_count,
            "stuck_checks": checks,
            "stuck_rate": stuck_count / checks if checks else 0.0,
            "accepted_worse": accepted_worse,
            "rejected": rejected,
            "last_state_value": current_value,
        },
    )