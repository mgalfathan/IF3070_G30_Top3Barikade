# python -m src.run_sa --seed 42 --out output/sa
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
from .evaluate.report import report


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
    parser.add_argument("problem", nargs="?", default="data/dataset.json")
    parser.add_argument("--t0", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objective", choices=["fee", "eta"], default="fee")
    parser.add_argument("--out", default="output/sa")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    result = simulated_annealing(problem, t0=args.t0, alpha=args.alpha, seed=args.seed,
                                 objective=args.objective)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(report(problem, result.initial_state, "Initial state", objective=args.objective))
    print(report(problem, result.final_state, "Final state", objective=args.objective))
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