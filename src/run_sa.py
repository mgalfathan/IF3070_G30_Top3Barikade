# python -m src.run_sa data/sample.json --seed 42 --out output/sa
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .algorithm.simulated_annealing import simulated_annealing
from .core.model import Problem, State
from .core.objective import CONFLICT, EMPTY, id_grid, summarize
from .evaluate.io import load_problem, save_result


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


def plot_history(history: dict, out_dir: Path) -> None:
    iters = range(1, len(history["value"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, history["value"], label="current", linewidth=1)
    ax.plot(iters, history["best"], label="best", linewidth=1.5)
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Objective function")
    ax.set_title("Simulated Annealing: nilai objective terhadap iterasi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sa_objective.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(iters, history["accept_prob"], s=2, alpha=0.5)
    ax.set_xlabel("Iterasi")
    ax.set_ylabel(r"$e^{\Delta E / T}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"Simulated Annealing: $e^{\Delta E / T}$ terhadap iterasi")
    fig.tight_layout()
    fig.savefig(out_dir / "sa_accept_prob.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated Annealing")
    parser.add_argument("problem", nargs="?", default="data/sample.json")
    parser.add_argument("--t0", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="output/sa")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    result = simulated_annealing(problem, t0=args.t0, alpha=args.alpha, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(report(problem, result.initial_state, "Initial state"))
    print(report(problem, result.final_state, "Final state"))
    print(f"Iterasi           : {result.iterations_run}")
    print(f"Durasi            : {result.duration:.3f} s")
    print(f"Frekuensi stuck   : {result.extra['stuck_count']} dari "
          f"{result.extra['stuck_checks']} pengecekan "
          f"({result.extra['stuck_rate']:.1%})")
    print(f"Terima lebih buruk: {result.extra['accepted_worse']}")
    print(f"Ditolak           : {result.extra['rejected']}")

    plot_history(result.history, out_dir)
    save_result(problem, result, out_dir / "result.json")
    print(f"\nPlot & hasil disimpan di {out_dir}/")


if __name__ == "__main__":
    main()