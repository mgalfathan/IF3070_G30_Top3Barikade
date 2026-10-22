# Run: python -m src.run_hc data/dataset.json --seed 1 --out output/hc/run1

import argparse

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .algorithm.hill_climbing import steepest_ascent
from .evaluate.io import load_problem, save_result
from .evaluate.report import report
from .core.objective import OBJECTIVES


# Membuat tampilan grafik history Nilai Objective terhadap Iterasi
def plot_history (history:dict, out_dir: Path):
    iters = range(1, len(history["value"]) + 1)

    fig, ax = plt.subplots(figsize = (9, 4))

    ax.plot(iters, history['value'], marker = 'o', markersize = 4)
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Nilai Objective")
    ax.set_title("Steepest Ascent Hill-Climbing: Nilai Objective terhadap Iterasi")
    fig.tight_layout()
    fig.savefig(out_dir/"steepest_hc.png", dpi= 150)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Steepest Ascent Hill-Climbing")
    parser.add_argument("problem", help="path file JSON permasalahan")
    parser.add_argument("--objective", choices=OBJECTIVES)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--p-placed", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="output/hc")
    args = parser.parse_args()

    problem = load_problem(args.problem)
    result = steepest_ascent(problem, max_iter=args.max_iter, p_placed=args.p_placed, seed=args.seed, objective=args.objective)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(report(problem, result.initial_state, "Initial State", objective=args.objective))
    print(report(problem, result.final_state, "Final State", objective=args.objective))
    print(f"Iterasi: {result.extra['iterations']}")
    print(f"Durasi: {result.duration:.3f} s")

    plot_history(result.history, out_dir)
    save_result(problem, result, out_dir / "result.json")
    print(f"\nPlot & hasil disimpan di {out_dir}")
    
if __name__ == "__main__":
    main()