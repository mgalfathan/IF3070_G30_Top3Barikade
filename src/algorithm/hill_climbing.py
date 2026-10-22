from __future__ import annotations
import random
from typing import Iterator
from ..core.model import Problem, State
from ..core.objective import evaluate
from ..evaluate.init import random_state
from ..evaluate.moves import move, move_out, rotate, swap
from ..evaluate.result import Result, Timer


#Fungsi bantuan untuk mengakses seluruh tetangga
def all_neighbors(problem: Problem, state: State) -> Iterator[State]:
    #tidak perlu dirotate karena jika diluar kapal tidak perlu dicek
    for i in state.placed_indices():
        yield rotate(state, i)
    for i in range(problem.n):
        vehicle = problem.vehicles[i]
        placement = state.placements[i]

        if placement.placed :
            yield move_out(state,i)

        if vehicle.fits(problem.ship, placement.orientation) != True:
            continue

        dx, dy = vehicle.dims(placement.orientation)

        for j in range(0, problem.ship.width - dx + 1):
            for k in range(0, problem.ship.length - dy + 1):
                if placement.placed and j == placement.x and k == placement.y:
                    continue
                else: 
                    yield move(state, i, j, k)

    for l in range(problem.n):
        for m in range(l+1, problem.n):
            placementl = state.placements[l]
            placementm = state.placements[m]

            if placementl.placed == False and placementm.placed == False:
                continue

            yield swap(state, l, m)
        
    
def steepest_ascent(problem: Problem,
                    initial: State | None = None,
                    max_iter: int = 1000,
                    p_placed: float = 0.5,
                    seed: int | None = None) -> Result:
    rng = random.Random(seed)

    if initial is not None:
        current = initial.copy() 
    else:
        current = random_state(problem, rng, p_placed)
    current_value = evaluate(problem, current)
    initial_state, initial_value = current.copy(), current_value

    history = {"value" : []}

    with Timer() as timer:
        for _ in range(max_iter):
            current_neighbor = current_value
            best_neighbor = None

            for neighbor in all_neighbors(problem, current):
                neighbor_value = evaluate(problem, neighbor)
                if current_neighbor < neighbor_value:
                    current_neighbor = neighbor_value
                    best_neighbor = neighbor

            if best_neighbor == None: 
                break
            else:
                current = best_neighbor
                current_value = current_neighbor
                history["value"].append(current_value) 

    return Result(
        algorithm = "steepest_ascent_hill_climbing",
        initial_state = initial_state,
        final_state = current,
        initial_value = initial_value,
        final_value = current_value,
        history= history,
        duration = timer.elapsed,
        params = {"max_iter": max_iter, "p_placed": p_placed, "seed" : seed}, 
        extra = {"iterations" : len(history["value"])}
    )
    