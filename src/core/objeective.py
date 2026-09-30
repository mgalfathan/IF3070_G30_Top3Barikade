from __future__ import annotations
import numpy as np
from .model import Placement, Problem, State, Vehicle

EMPTY = -1
CONFLICT = -2  

def footprint(vehicle: Vehicle, p: Placement) -> tuple[int, int]:
    """Ukuran (dx, dy) kendaraan sesuai orientasi pada placement."""
    return vehicle.dims(p.orientation)


def occupied_cells(vehicle: Vehicle, p: Placement) -> list[tuple[int, int]]:
    """Daftar sel (x, y) yang ditempati kendaraan (termasuk yang di luar geladak)."""
    dx, dy = footprint(vehicle, p)
    return [(x, y) for y in range(p.y, p.y + dy) for x in range(p.x, p.x + dx)]


def _clipped_box(problem: Problem, vehicle: Vehicle, p: Placement):
    """Kotak (x0, y0, x1, y1) bagian kendaraan di dalam geladak + jumlah sel di luar."""
    ship = problem.ship
    dx, dy = footprint(vehicle, p)
    x0, y0 = max(p.x, 0), max(p.y, 0)
    x1, y1 = min(p.x + dx, ship.width), min(p.y + dy, ship.length)
    inside = (x1 - x0) * (y1 - y0) if (x0 < x1 and y0 < y1) else 0
    return (x0, y0, x1, y1), dx * dy - inside


def build_grid(problem: Problem, state: State) -> tuple[np.ndarray, int]:
    """
    Grid hitungan berukuran (length, width); grid[y, x] = banyak kendaraan
    yang menempati sel (x, y). Juga mengembalikan jumlah sel di luar geladak.
    """
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
    """
    Grid untuk visualisasi: indeks kendaraan pada tiap sel,
    EMPTY (-1) jika kosong, CONFLICT (-2) jika overlap.
    """
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


def violations(problem: Problem, state: State) -> dict:
    """Besar tiap pelanggaran constraint (0 semua berarti state valid)."""
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


def evaluate(problem: Problem, state: State, penalty: float | None = None) -> float:
    """Fungsi objektif yang dimaksimalkan oleh semua algoritma."""
    fee = total_fee(problem, state)
    viol = violations(problem, state)
    total_violation = viol["overlap"] + viol["out_of_bounds"] + viol["overweight"]
    if total_violation == 0:
        return fee
    p = default_penalty(problem) if penalty is None else penalty
    return fee - p * (1 + total_violation)


def summarize(problem: Problem, state: State) -> dict:
    """Ringkasan lengkap state untuk laporan dan visualisasi."""
    grid, _ = build_grid(problem, state)
    viol = violations(problem, state)
    return {
        "value": evaluate(problem, state),
        "total_fee": total_fee(problem, state),
        "total_weight": total_weight(problem, state),
        "max_capacity": problem.ship.max_capacity,
        "valid": all(val == 0 for val in viol.values()),
        "violations": viol,
        "placed_count": len(state.placed_indices()),
        "vehicle_count": problem.n,
        "deck_utilization": float(np.count_nonzero(grid)) / problem.ship.area,
    }