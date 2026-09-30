from __future__ import annotations
from dataclasses import dataclass, replace

HORIZONTAL = "H"
VERTICAL = "V"
ORIENTATIONS = (HORIZONTAL, VERTICAL)


@dataclass(frozen=True)
class Vehicle:
    """Data kendaraan yang tidak pernah berubah selama pencarian."""
    id: str
    w: int
    l: int
    fee: float
    weight: float
    eta: int = 0

    @property
    def area(self) -> int:
        return self.w * self.l

    def dims(self, orientation: str) -> tuple[int, int]:
        """Ukuran (dx, dy) kendaraan pada orientasi tertentu."""
        if orientation == HORIZONTAL:
            return self.w, self.l
        if orientation == VERTICAL:
            return self.l, self.w
        raise ValueError(f"Orientasi tidak dikenal: {orientation!r}")

    def fits(self, ship: Ship, orientation: str) -> bool:
        """Apakah kendaraan pada orientasi ini bisa muat di geladak kosong."""
        dx, dy = self.dims(orientation)
        return dx <= ship.width and dy <= ship.length


@dataclass(frozen=True)
class Ship:
    width: int
    length: int
    max_capacity: float

    @property
    def area(self) -> int:
        return self.width * self.length


@dataclass(frozen=True)
class Problem:
    """Satu instance permasalahan: satu kapal + daftar kendaraan."""
    ship: Ship
    vehicles: tuple[Vehicle, ...]

    def __post_init__(self) -> None:
        if self.ship.width <= 0 or self.ship.length <= 0:
            raise ValueError("Dimensi geladak harus positif.")
        ids = [v.id for v in self.vehicles]
        if len(ids) != len(set(ids)):
            raise ValueError("Setiap kendaraan harus memiliki id yang unik.")
        for v in self.vehicles:
            if v.w <= 0 or v.l <= 0:
                raise ValueError(f"Dimensi kendaraan {v.id} harus positif.")

    @property
    def n(self) -> int:
        return len(self.vehicles)

    def index_of(self, vehicle_id: str) -> int:
        for i, v in enumerate(self.vehicles):
            if v.id == vehicle_id:
                return i
        raise KeyError(vehicle_id)


@dataclass
class Placement:
    """Keputusan penempatan untuk satu kendaraan (bagian yang berubah-ubah)."""
    placed: bool = False
    x: int = 0
    y: int = 0
    orientation: str = HORIZONTAL


@dataclass
class State:
    """
    Satu solusi (state). placements[i] adalah penempatan untuk
    problem.vehicles[i], sehingga urutannya harus selalu sama.
    """
    placements: list[Placement]

    def copy(self) -> State:
        """Deep copy. Selalu gunakan ini sebelum mengubah state."""
        return State([replace(p) for p in self.placements])

    def placed_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.placements) if p.placed]

    def unplaced_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.placements) if not p.placed]

    def __len__(self) -> int:
        return len(self.placements)