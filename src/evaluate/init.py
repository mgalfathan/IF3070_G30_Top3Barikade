from __future__ import annotations
import random
from ..core.model import ORIENTATIONS, Placement, Problem, Ship, State, Vehicle

def random_position(vehicle: Vehicle, ship: Ship, orientation: str,
                    rng: random.Random) -> tuple[int, int]:
    dx, dy = vehicle.dims(orientation)
    if dx > ship.width or dy > ship.length:
        raise ValueError(f"Kendaraan {vehicle.id} tidak muat pada orientasi {orientation}.")
    return rng.randint(0, ship.width - dx), rng.randint(0, ship.length - dy)


def random_placement(vehicle: Vehicle, ship: Ship, rng: random.Random,
                     p_placed: float = 0.5) -> Placement:
    feasible = [o for o in ORIENTATIONS if vehicle.fits(ship, o)]
    if not feasible:
        return Placement(placed=False)
    orientation = rng.choice(feasible)
    x, y = random_position(vehicle, ship, orientation, rng)
    return Placement(placed=rng.random() < p_placed, x=x, y=y, orientation=orientation)


def random_state(problem: Problem, rng: random.Random | None = None,
                 p_placed: float = 0.5) -> State:
    # p_placed = peluang kendaraan dimasukkan ke kapal, makin kecil makin jarang overlap
    rng = rng or random.Random()
    return State([random_placement(v, problem.ship, rng, p_placed) for v in problem.vehicles])