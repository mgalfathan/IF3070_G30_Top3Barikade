from __future__ import annotations
 
from ..core.model import Problem, State
from ..core.objective import CONFLICT, EMPTY, id_grid, summarize

def deck_text(problem: Problem, state: State) -> str:
    # '.' = kosong, '#' = overlap
    grid = id_grid(problem, state)
    rows = []
    for row in grid:
        cells = []
        for c in row:
            if c == EMPTY:
                cells.append(".")
            elif c == CONFLICT:
                cells.append("#")
            else:
                cells.append(problem.vehicles[c].id[0])
        rows.append(" ".join(cells))
    return "\n".join(rows)


def vehicles_text(problem: Problem, state: State) -> str:
    lines = [f"{'id':<4}{'ukuran':<8}{'fee':>8}{'berat':>8}  status"]
    for v, p in zip(problem.vehicles, state.placements):
        status = f"di kapal (x={p.x}, y={p.y}, {p.orientation})" if p.placed else "di luar"
        lines.append(f"{v.id:<4}{f'{v.w}x{v.l}':<8}{v.fee:>8g}{v.weight:>8g}  {status}")
    return "\n".join(lines)


def report(problem: Problem, state: State, title: str) -> str:
    s = summarize(problem, state)
    return (f"===== {title} =====\n"
            f"{vehicles_text(problem, state)}\n\n"
            f"Geladak ({problem.ship.width}x{problem.ship.length}):\n"
            f"{deck_text(problem, state)}\n\n"
            f"Value          : {s['value']:g}\n"
            f"Total fee      : {s['total_fee']:g}\n"
            f"Total berat    : {s['total_weight']:g} / {s['max_capacity']:g}\n"
            f"Valid          : {s['valid']} {s['violations']}\n"
            f"Kendaraan masuk: {s['placed_count']} / {s['vehicle_count']}\n"
            f"Utilisasi deck : {s['deck_utilization']:.1%}\n")