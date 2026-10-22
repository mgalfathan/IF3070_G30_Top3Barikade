from __future__ import annotations
import numpy as np
from .model import Placement, Problem, State, Vehicle

EMPTY = -1
CONFLICT = -2
OBJECTIVES = ("fee", "eta")

def footprint(vehicle: Vehicle, p: Placement) -> tuple[int, int]:
    return vehicle.dims(p.orientation)


def occupied_cells(vehicle: Vehicle, p: Placement) -> list[tuple[int, int]]:
    # termasuk sel yang berada di luar geladak
    dx, dy = footprint(vehicle, p)
    return [(x, y) for y in range(p.y, p.y + dy) for x in range(p.x, p.x + dx)]


def _clipped_box(problem: Problem, vehicle: Vehicle, p: Placement):
    # bagian kendaraan yang ada di dalam geladak + jumlah sel yang keluar batas
    ship = problem.ship
    dx, dy = footprint(vehicle, p)
    x0, y0 = max(p.x, 0), max(p.y, 0)
    x1, y1 = min(p.x + dx, ship.width), min(p.y + dy, ship.length)
    inside = (x1 - x0) * (y1 - y0) if (x0 < x1 and y0 < y1) else 0
    return (x0, y0, x1, y1), dx * dy - inside


def build_grid(problem: Problem, state: State) -> tuple[np.ndarray, int]:
    # grid[y, x] = jumlah kendaraan yang menempati sel (x, y)
    ship = problem.ship
    grid = np.zeros((ship.length, ship.width), dtype=np.int32)
    out_of_bounds = 0
    for v, p in zip(problem.vehicles, state.placements):
        if not p.placed:
            continue
        (x0, y0, x1, y1), outside = _clipped_box(problem, v, p)
        if x0 < x1 and y0 < y1:
            grid[y0:y1, x0:x1] += 1
        out_of_bounds += outside
    return grid, out_of_bounds


def id_grid(problem: Problem, state: State) -> np.ndarray:
    # isi sel: indeks kendaraan, EMPTY jika kosong, CONFLICT jika overlap
    ship = problem.ship
    grid = np.full((ship.length, ship.width), EMPTY, dtype=np.int32)
    for i, (v, p) in enumerate(zip(problem.vehicles, state.placements)):
        if not p.placed:
            continue
        (x0, y0, x1, y1), _ = _clipped_box(problem, v, p)
        if x0 < x1 and y0 < y1:
            region = grid[y0:y1, x0:x1]
            region[region == EMPTY] = i
            region[(region != i) & (region != EMPTY)] = CONFLICT
    return grid


def total_fee(problem: Problem, state: State) -> float:
    return sum(v.fee for v, p in zip(problem.vehicles, state.placements) if p.placed)


def total_weight(problem: Problem, state: State) -> float:
    return sum(v.weight for v, p in zip(problem.vehicles, state.placements) if p.placed)


def eta_range(problem: Problem) -> tuple[int, int]:
    # ETA terkecil dan terbesar dari seluruh kendaraan di problem
    etas = [v.eta for v in problem.vehicles]
    return min(etas), max(etas)


def eta_weight(vehicle: Vehicle, eta_min: int, eta_max: int) -> float:
    # Menghitung bobot urgensi tiap kendaraan, ETA kecil memberikan nilai lebih baik
    return (eta_max - vehicle.eta + 1) / (eta_max - eta_min + 1)


def total_eta_score(problem: Problem, state: State) -> float:
    # total fee x bobot ETA kendaraan di kapal
    eta_min, eta_max = eta_range(problem)
    return sum(v.fee * eta_weight(v, eta_min, eta_max)
               for v, p in zip(problem.vehicles, state.placements) if p.placed)


def base_value(problem: Problem, state: State, objective: str = "fee") -> float:
    # nilai dasar sebelum dikenakan penalti
    if objective == "fee":
        return total_fee(problem, state)
    if objective == "eta":
        return total_eta_score(problem, state)
    raise ValueError(f"objective tidak dikenal: {objective!r} (pilihan: {OBJECTIVES})")


def violations(problem: Problem, state: State) -> dict:
    grid, out_of_bounds = build_grid(problem, state)
    overlap = int(np.sum(grid[grid > 1] - 1))
    overweight = max(0.0, total_weight(problem, state) - problem.ship.max_capacity)
    return {
        "overlap": overlap,
        "out_of_bounds": int(out_of_bounds),
        "overweight": float(overweight),
    }


def is_valid(problem: Problem, state: State) -> bool:
    return all(val == 0 for val in violations(problem, state).values())


def default_penalty(problem: Problem) -> float:
    return sum(v.fee for v in problem.vehicles) + 1.0


def evaluate(problem: Problem, state: State, penalty: float | None = None, objective: str = "fee") -> float:
    # valid: total fee; tidak valid: fee - penalty * (1 + total pelanggaran)
    value = base_value(problem, state, objective)
    viol = violations(problem, state)
    total_violation = viol["overlap"] + viol["out_of_bounds"] + viol["overweight"]
    if total_violation == 0:
        return value
    p = default_penalty(problem) if penalty is None else penalty
    return value - p * (1 + total_violation)


def summarize(problem: Problem, state: State, objective: str = "fee") -> dict:
    grid, _ = build_grid(problem, state)
    viol = violations(problem, state)
    return {
        "objective": objective,
        "value": evaluate(problem, state, objective=objective),
        "total_fee": total_fee(problem, state),
        "total_eta_score": total_eta_score(problem, state),
        "total_weight": total_weight(problem, state),
        "max_capacity": problem.ship.max_capacity,
        "valid": all(val == 0 for val in viol.values()),
        "violations": viol,
        "placed_count": len(state.placed_indices()),
        "vehicle_count": problem.n,
        "deck_utilization": float(np.count_nonzero(grid)) / problem.ship.area,
    }