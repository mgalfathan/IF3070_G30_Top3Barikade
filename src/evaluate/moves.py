from __future__ import annotations
import random
from .init import random_position
from .model import HORIZONTAL, VERTICAL, Problem, State

MOVE_TYPES = ("swap", "move", "rotate")


def swap(state: State, i: int, j: int) -> State:
    """
    Tukar placement kendaraan i dan j (posisi, orientasi, status dalam/luar).
    Jika satu di dalam dan satu di luar kapal, efeknya mengganti kendaraan
    yang diangkut. Karena ukuran bisa berbeda, hasilnya bisa overlap/keluar batas.
    """
    new = state.copy()
    new.placements[i], new.placements[j] = new.placements[j], new.placements[i]
    return new


def move(state: State, i: int, x: int, y: int, placed: bool = True) -> State:
    """Pindahkan kendaraan i ke (x, y). placed=False berarti dipindah ke luar kapal."""
    new = state.copy()
    p = new.placements[i]
    p.x, p.y, p.placed = x, y, placed
    return new


def move_out(state: State, i: int) -> State:
    """Pindahkan kendaraan i ke luar kapal (kasus khusus dari move)."""
    p = state.placements[i]
    return move(state, i, p.x, p.y, placed=False)


def rotate(state: State, i: int) -> State:
    """Ubah orientasi kendaraan i (H <-> V) dengan titik kiri-atas tetap."""
    new = state.copy()
    p = new.placements[i]
    p.orientation = VERTICAL if p.orientation == HORIZONTAL else HORIZONTAL
    return new


def random_neighbor(problem: Problem, state: State, rng: random.Random,
                    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
                    p_move_out: float = 0.1) -> State:
    """
    Bangkitkan satu tetangga acak. Dipakai oleh Simulated Annealing,
    Stochastic Hill Climbing, dan operator mutasi Genetic Algorithm.

    weights    : bobot pemilihan (swap, move, rotate)
    p_move_out : peluang move mengeluarkan kendaraan yang sedang di kapal
    """
    n = problem.n
    placed = state.placed_indices()
    kind = rng.choices(MOVE_TYPES, weights=weights)[0]

    if kind == "swap" and n >= 2 and placed:
        i = rng.choice(placed)
        j = rng.choice([k for k in range(n) if k != i])
        return swap(state, i, j)

    if kind == "rotate" and placed:
        return rotate(state, rng.choice(placed))

    i = rng.randrange(n)
    vehicle, p = problem.vehicles[i], state.placements[i]
    if (p.placed and rng.random() < p_move_out) or not vehicle.fits(problem.ship, p.orientation):
        return move_out(state, i) if p.placed else state.copy()
    x, y = random_position(vehicle, problem.ship, p.orientation, rng)
    return move(state, i, x, y, placed=True)