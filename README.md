# IF3070_G30_Top3Barikade

Tugas Besar 1 IF3070 Dasar Inteligensi Artifisial.

## Deskripsi

Program ini mencari penempatan kendaraan pada geladak kapal kargo milik PT Intim Line. Solusi menentukan kendaraan mana yang diangkut, serta posisi dan orientasi (horizontal atau vertikal) setiap kendaraan di geladak. Tujuannya memaksimalkan total ShippingFee dengan tetap memenuhi tiga batasan: kendaraan tidak saling bertumpuk, seluruh kendaraan berada di dalam batas geladak, dan total berat tidak melebihi kapasitas kapal.

Algoritma local search yang diimplementasikan:

- Steepest Ascent Hill-Climbing
- Simulated Annealing 
- Genetic Algorithm 

Pelanggaran batasan akan diberikan penalti pada objective function, sehingga algoritma dapat melewati state yang belum valid. 

## Struktur Repository

```
.
├── data/
│   └── dataset.json            instance masalah
├── docs/
├── output/                     hasil run (plot dan result.json)
│   ├── hc/
│   ├── sa/
│   └── ga/
└── src/
    ├── core/
    │   ├── model.py            Vehicle, Ship, Problem, Placement, State
    │   └── objective.py        grid, pelanggaran constraint, evaluate
    ├── evaluate/
    │   ├── init.py             initial state acak
    │   ├── moves.py            swap, move, rotate, random_neighbor
    │   ├── io.py               baca instance dan simpan hasil
    │   ├── result.py           Result dan Timer
    │   └── report.py           visualisasi state dalam teks
    ├── algorithm/
    │   ├── hill_climbing.py
    │   ├── simulated_annealing.py
    │   └── genetic_algorithm.py
    ├── run_hc.py               program utama Hill-Climbing
    ├── run_sa.py               program utama Simulated Annealing
    └── run_ga.py               program utama Genetic Algorithm
```

## Setup

Kebutuhan: Python 3.10 atau lebih baru.

```bash
git clone https://github.com/mgalfathan/IF3070_G30_Top3Barikade.git
cd IF3070_G30_Top3Barikade
pip install numpy matplotlib
```

## Cara Menjalankan

Semua perintah dijalankan dari root repository.

### Hill-Climbing

```bash
python -m src.run_hc --seed 1 --out output/hc/run1
```

| Argumen | Default | Keterangan |
|---|---|---|
| `problem` | `data/dataset.json`  | path file JSON permasalahan |
| `--max-iter` | 100 | batas iterasi |
| `--p-placed` | 0.5 | peluang kendaraan masuk kapal pada state awal |
| `--seed` | acak | seed random |
| `--out` | `output/hc` | folder output |

### Simulated Annealing

```bash
python -m src.run_sa --seed 1 --out output/sa/sa_1
```

Tanpa argumen file, program memakai `data/dataset.json`.

| Argumen | Default | Keterangan |
|---|---|---|
| `problem` | `data/dataset.json` | path file JSON permasalahan |
| `--t0` | dihitung dari data | suhu awal |
| `--alpha` | 0.995 | faktor pendinginan geometris |
| `--seed` | acak | seed random |
| `--out` | `output/sa` | folder output |

### Genetic Algorithm

```bash
python -m src.run_ga --pop 50 --iter 300 --seed 1 --out output/ga/run1
```

| Argumen | Default | Keterangan |
|---|---|---|
| `problem` | `data/dataset.json`  | path file JSON permasalahan |
| `--pop` | 50 | jumlah populasi |
| `--iter` | 300 | banyak iterasi (generasi) |
| `--crossover-rate` | 0.8 | peluang crossover |
| `--mutation-rate` | 0.2 | peluang mutasi |
| `--elite` | 2 | jumlah individu elit |
| `--tournament` | 3 | ukuran turnamen |
| `--p-placed` | 0.5 | peluang kendaraan masuk kapal pada populasi awal |
| `--crossover` | `uniform` | `uniform` atau `spatial` |
| `--seed` | acak | seed random |
| `--out` | `output/ga` | folder output |

Setiap program mencetak state awal dan state akhir (tabel kendaraan, grid geladak, dan value), lalu menyimpan plot dan `result.json` ke folder output.

## Format Input

Instance masalah berupa file JSON, contohnya `data/dataset.json`:

```json
{
  "ship": {"width": 8, "length": 10, "max_capacity": 30},
  "vehicles": [
    {"id": "A", "w": 2, "l": 4, "fee": 120, "weight": 4},
    {"id": "B", "w": 2, "l": 5, "fee": 150, "weight": 5}
  ]
}
```

| Field | Keterangan |
|---|---|
| `ship.width`, `ship.length` | ukuran geladak (satuan sel grid) |
| `ship.max_capacity` | kapasitas berat maksimum |
| `vehicles[].id` | identifier unik kendaraan |
| `vehicles[].w`, `vehicles[].l` | ukuran kendaraan (satuan sel grid) |
| `vehicles[].fee` | ShippingFee (juta rupiah) |
| `vehicles[].weight` | berat kendaraan (unit) |

## Pembagian Tugas

| Nama | NIM | Kontribusi |
|---|---|---|
| Reza Nashwan Faizi | 18224008 | Steepest Ascent Hill-Climbing (algoritma, membuat seluruh tetangga, program utama, eksperimen), penyusunan laporan |
| Aldyto Rafif Abhinanya | 18224043 | (isi) |
| Muhammad Ghiffary Alfathan |18224084 |Simulated Annealing (algoritma, deteksi stuck, program utama, eksperimen), dataset, penyusunan laporan|
