from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..core.model import Placement, Problem, Ship, State, Vehicle
from ..core.objective import summarize
from .result import Result

def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def problem_from_dict(data: dict) -> Problem:
    s = data["ship"]
    ship = Ship(width=int(s["width"]), length=int(s["length"]),
                max_capacity=float(s["max_capacity"]))
    vehicles = tuple(
        Vehicle(id=str(v["id"]), w=int(v["w"]), l=int(v["l"]),
                fee=float(v["fee"]), weight=float(v["weight"]),
                eta=int(v.get("eta", 0)))
        for v in data["vehicles"]
    )
    return Problem(ship=ship, vehicles=vehicles)

def problem_to_dict(problem: Problem) -> dict:
    s = problem.ship
    return {
        "ship": {"width": s.width, "length": s.length, "max_capacity": s.max_capacity},
        "vehicles": [
            {"id": v.id, "w": v.w, "l": v.l, "fee": v.fee, "weight": v.weight, "eta": v.eta}
            for v in problem.vehicles
        ],
    }

def load_problem(path: str | Path) -> Problem:
    return problem_from_dict(load_json(path))

def state_to_dict(problem: Problem, state: State, include_summary: bool = True) -> dict:
    data: dict[str, Any] = {
        "vehicles": [
            {"id": v.id, "placed": p.placed, "x": p.x, "y": p.y, "orientation": p.orientation}
            for v, p in zip(problem.vehicles, state.placements)
        ]
    }
    if include_summary:
        data["summary"] = summarize(problem, state)
    return data


def state_from_dict(problem: Problem, data: dict) -> State:
    # kendaraan yang tidak ada di data dianggap di luar kapal
    by_id = {item["id"]: item for item in data["vehicles"]}
    placements = []
    for v in problem.vehicles:
        item = by_id.get(v.id)
        if item is None:
            placements.append(Placement(placed=False))
        else:
            placements.append(Placement(placed=bool(item["placed"]), x=int(item["x"]),
                                        y=int(item["y"]), orientation=item["orientation"]))
    return State(placements)

def result_to_dict(problem: Problem, result: Result) -> dict:
    return {
        "algorithm": result.algorithm,
        "params": result.params,
        "duration": result.duration,
        "iterations_run": result.iterations_run,
        "initial_value": result.initial_value,
        "final_value": result.final_value,
        "initial_state": state_to_dict(problem, result.initial_state),
        "final_state": state_to_dict(problem, result.final_state),
        "history": result.history,
        "extra": result.extra,
    }

def save_result(problem: Problem, result: Result, path: str | Path) -> None:
    save_json(result_to_dict(problem, result), path)

def load_result(problem: Problem, path: str | Path) -> Result:
    data = load_json(path)
    return Result(
        algorithm=data["algorithm"],
        initial_state=state_from_dict(problem, data["initial_state"]),
        final_state=state_from_dict(problem, data["final_state"]),
        initial_value=data["initial_value"],
        final_value=data["final_value"],
        history=data.get("history", {}),
        duration=data.get("duration", 0.0),
        params=data.get("params", {}),
        extra=data.get("extra", {}),
    )