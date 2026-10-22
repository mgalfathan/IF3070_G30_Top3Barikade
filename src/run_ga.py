# Contoh run:
#   py -m src.run_ga data/sample.json --pop 50 --iter 300 --seed 42
#   py -m src.run_ga data/sample.json --pop 50 --iter 300 --seed 42 --crossover spatial --out output/ga_spatial

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .algorithm.genetic_algorithm import genetic_algorithm
from .core.model import Problem
from .evaluate.io import load_problem, save_result
from .evaluate.result import Result
from .evaluate.report import report


def plot_history(result: Result, out_dir: Path) -> Path:
    """
    Plot nilai objective maksimum dan rata-rata populasi terhadap generasi.
    """
    history = result.history
    gens = range(1, len(history["max"]) + 1)
    p = result.params

    fig, (ax_max, ax_avg) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_max.plot(gens, history["max"], color="tab:blue", linewidth=1.5)
    ax_max.set_ylabel("Fitness maksimum")
    ax_max.grid(alpha=0.3)

    ax_avg.plot(gens, history["avg"], color="tab:orange", linewidth=1.2)
    ax_avg.set_ylabel("Fitness rata-rata")
    ax_avg.set_xlabel("Generasi (iterasi)")
    ax_avg.grid(alpha=0.3)

    fig.suptitle(f"Genetic Algorithm: objective terhadap iterasi\n"
                 f"populasi={p['population_size']}, iterasi={p['iterations']}, "
                 f"crossover={p['crossover_type']}, seed={p['seed']}")
    fig.tight_layout()

    path = out_dir / "ga_objective.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary_text(result: Result) -> str:
    """Ringkasan eksperimen."""
    p = result.params
    return (f"===== Ringkasan eksperimen =====\n"
            f"Objective awal       : {result.initial_value:g}\n"
            f"Objective akhir      : {result.final_value:g}\n"
            f"Final state valid    : {result.extra['final_valid']}\n"
            f"Terbaik di generasi  : {result.extra['best_generation']}\n"
            f"Jumlah populasi      : {p['population_size']}\n"
            f"Banyak iterasi       : {p['iterations']}\n"
            f"Crossover rate       : {p['crossover_rate']}\n"
            f"Mutation rate        : {p['mutation_rate']}\n"
            f"Jumlah elit          : {p['elite_count']}\n"
            f"Ukuran turnamen      : {p['tournament_size']}\n"
            f"p_placed             : {p['p_placed']}\n"
            f"Jenis crossover      : {p['crossover_type']}\n"
            f"Seed                 : {p['seed']}\n"
            f"Durasi pencarian     : {result.duration:.3f} s\n")


def run_once(problem: Problem, params: dict, out_dir: str | Path,
             verbose: bool = True) -> Result:
    """
    Menjalankan satu kali GA lalu menyimpan report.txt, ga_objective.png,
    dan result.json ke out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = genetic_algorithm(problem, **params)

    text = (report(problem, result.initial_state, "Initial state") + "\n"
            + report(problem, result.final_state, "Final state") + "\n"
            + summary_text(result))
    (out_dir / "report.txt").write_text(text, encoding="utf-8")

    plot_history(result, out_dir)
    save_result(problem, result, out_dir / "result.json")

    if verbose:
        print(text)
        print(f"Plot & hasil disimpan di {out_dir}/")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Genetic Algorithm")
    parser.add_argument("problem", help="path file JSON permasalahan")
    parser.add_argument("--pop", type=int, default=50, help="jumlah populasi")
    parser.add_argument("--iter", type=int, default=300, help="banyak iterasi (generasi)")
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elite", type=int, default=2, help="jumlah individu elit")
    parser.add_argument("--tournament", type=int, default=3, help="ukuran turnamen")
    parser.add_argument("--p-placed", type=float, default=0.5,
                        help="peluang kendaraan masuk kapal pada populasi awal")
    parser.add_argument("--crossover", choices=["uniform", "spatial"], default="uniform",
                        help="jenis crossover (default: uniform)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="output/ga")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    params = {
        "population_size": args.pop,
        "iterations": args.iter,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "elite_count": args.elite,
        "tournament_size": args.tournament,
        "p_placed": args.p_placed,
        "crossover_type": args.crossover,
        "seed": args.seed,
    }
    run_once(problem, params, args.out)


if __name__ == "__main__":
    main()