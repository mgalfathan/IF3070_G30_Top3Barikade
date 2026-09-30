from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

from .model import State


@dataclass
class Result:
    algorithm: str
    initial_state: State
    final_state: State
    initial_value: float
    final_value: float
    history: dict[str, list[float]] = field(default_factory=dict)
    duration: float = 0.0               # detik
    params: dict[str, Any] = field(default_factory=dict) 
    extra: dict[str, Any] = field(default_factory=dict) 

    @property
    def iterations_run(self) -> int:
        return max((len(v) for v in self.history.values()), default=0)


class Timer:
    """
    Pengukur durasi proses pencarian.

        with Timer() as t:
            ...  # proses pencarian
        durasi = t.elapsed
    """

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start