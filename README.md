## FASTRGG Python Port

This repository contains a **pure Python** implementation of the original
CUDA-based FASTRGG random graph generator for Erdős–Rényi graphs \(G(n, p)\).

The Python code lives in the `fastgg` package and is intended to be:

- **Easy to install**: standard Python package, no CUDA required.
- **Simple to run**: one CLI command or a small function call.
- **Faithful in spirit** to the original algorithms:
  - PER: independent Bernoulli trials per potential edge.
  - PZER: geometric-skip approach across all potential edges.
  - PPreZER: geometric-skip with a short precomputed skip table.


### 1. Requirements

- **Python**: 3.9 or newer.
- **OS**: any platform where Python 3.9+ is available (Linux, macOS, Windows).
- **No GPU / CUDA is required**. This is a CPU-only implementation.


### 2. Installation

There are two main ways to use the Python implementation:

#### 2.1. Run directly from the cloned repository

From the repo root (`fastgg`):

```bash
python -m fastgg.cli -n 1000 -p 0.1 -r 1 -a per
```

This uses the package in-place and **does not** install anything system‑wide.


#### 2.2. Install as a local package (recommended)

From the repo root:

```bash
python -m pip install .
```

After that you get a `fastgg` command on your `PATH`:

```bash
fastgg -n 1000 -p 0.1 -r 3 -a pzer
```


### 3. Conceptual Model

The Python implementation generates **directed** Erdős–Rényi graphs over
an \(n \times n\) adjacency matrix:

- Vertices are labeled `0, 1, ..., n-1`.
- Each ordered pair \((u, v)\) is a potential edge.
- Self-loops \((u, u)\) are allowed by design, mirroring the effective
  behavior of the original CUDA code where `TotalE = v^2`.

Internally, each edge is represented first as a **linear index**
`idx in [0, n*n)`, which can be mapped to `(u, v)` as:

- `u = idx // n`
- `v = idx % n`

The high-level distribution is always G(n, p) over this directed edge set,
but the three algorithms differ in how they *enumerate* present edges.


### 4. Command-Line Usage

Once installed (or from the repo via `python -m fastgg.cli`), the CLI is:

```bash
fastgg -n N -p P [-r RUNS] [-s SEED] [-a ALGO] [--log-csv PATH] [--edges-out PATH]
```

#### 4.1. Required arguments

- **`-n, --vertices N`**:  
  Number of vertices \(n\).

- **`-p, --probability P`**:  
  Edge probability \(p \in [0, 1]\).


#### 4.2. Optional arguments

- **`-r, --runs RUNS`** (default: `1`):  
  How many independent runs to perform.  
  Each run uses seed `SEED + run_index`.

- **`-s, --seed SEED`** (default: `7`):  
  Base random seed. Useful for reproducibility.

- **`-a, --algorithm ALGO`** (default: `per`):  
  Which algorithm to use; allowed values:
  - `per` – PER algorithm: independent Bernoulli trial for each potential edge.
  - `pzer` – PZER algorithm: geometric skip between present edges.
  - `pprezer` – PPreZER algorithm: geometric skip with a short precomputed table.

- **`--log-csv PATH`** (optional):  
  Append **per‑run statistics** to the given CSV file. If the file does not
  exist it is created with a header. Each row has:
  - `run_index` – 0‑based index of the run.
  - `n` – number of vertices.
  - `p` – edge probability.
  - `algorithm` – `per`, `pzer`, or `pprezer`.
  - `seed` – seed actually used for that run.
  - `elapsed_ms` – total runtime in milliseconds (Python measurement).
  - `edge_count` – number of edges generated in that run.

- **`--edges-out PATH`** (optional):  
  Write **all edges from the last run** into a CSV file.  
  The file has a header `u,v` and one directed edge per row, with
  `0 <= u, v < n`.


#### 4.3. Example CLI invocations

- **Single run using PER, print stats only**:

  ```bash
  fastgg -n 1000 -p 0.05 -a per
  ```

- **Three runs using PZER, log stats to CSV**:

  ```bash
  fastgg -n 5000 -p 0.001 -r 3 -a pzer --log-csv results_pzer.csv
  ```

- **Generate a graph and save edges to a file**:

  ```bash
  fastgg -n 2000 -p 0.01 -a pprezer --edges-out edges_pprezer.csv
  ```

  After this, `edges_pprezer.csv` will contain:

  ```text
  u,v
  0,17
  0,832
  1,5
  ...
  ```


### 5. Programmatic Usage (Python API)

You can also call the generator directly from Python code.

#### 5.1. Basic usage

```python
from fastgg import Algorithm, generate_graph

result = generate_graph(
    n=1000,
    p=0.1,
    algorithm=Algorithm.PER,  # or Algorithm.PZER / Algorithm.PPREZER
    seed=42,
)

print("Edges:", result.edge_count)
print("Runtime (ms):", result.elapsed_ms)

# Get edges as (u, v) pairs
edges = result.as_uv_pairs()
print("First 5 edges:", edges[:5])
```


#### 5.2. Multiple runs with different seeds

```python
from fastgg import Algorithm, generate_graph

n = 500
p = 0.02
algo = Algorithm.PZER

for i in range(5):
    res = generate_graph(n=n, p=p, algorithm=algo, seed=100 + i)
    print(f"Run {i}: seed={res.seed}, edges={res.edge_count}, ms={res.elapsed_ms:.2f}")
```


### 6. Algorithm Details (Python vs. Original CUDA)

- **Random generator**:
  - The **PER** implementation uses a linear congruential generator (LCG)
    with the **same modulus and multiplier** as the original CUDA code
    (`M = 2147483647`, `A = 16807`), so its behavior is closely related,
    though not bit‑identical to the full GPU execution.
  - **PZER** and **PPreZER** use Python's `random.Random` seeded with the
    given seed, sampling geometric skips that are distribution‑equivalent
    to G(n, p) over the `n*n` potential directed edges.

- **Edge set**:
  - The original CUDA code effectively treats `v^2` positions as potential
    edges. The Python port maintains that convention.
  - If you require a *simple, undirected* G(n, p) without self‑loops,
    you can post‑process `result.as_uv_pairs()` to discard `(u, v)` where
    `u == v` and, for example, keep only edges with `u < v`.

- **Performance**:
  - This Python code is **not** a GPU implementation and is intended for
    research, prototyping, and moderate‑sized graphs.
  - For very large graphs (e.g. tens of millions of edges), prefer the
    original CUDA implementation or re‑implement the same algorithms using
    a GPU‑accelerated Python stack such as CuPy or Numba.


### 7. How to Replace the Original Binary in Workflows

If you previously used the CUDA binary like this:

```bash
FastGG.exe -n 1000 -p 0.1 -r 1 -a 1
```

you can conceptually replace it with the Python CLI:

```bash
fastgg -n 1000 -p 0.1 -r 1 -a per --edges-out edges.csv
```

Key differences:

- The **Python CLI prints timing and edge counts to stdout**, and optionally
  writes per‑run statistics to a configurable CSV via `--log-csv`.
- **Edges are always written as `u,v` pairs** in a CSV when `--edges-out`
  is provided; you no longer have to parse binary buffers or infer formats.


### 8. Troubleshooting

- **`ModuleNotFoundError: No module named 'fastgg'`**  
  - Ensure you are either:
    - running from the repo root with `python -m fastgg.cli ...`, **or**
    - have installed the package via `python -m pip install .`.

- **CLI command `fastgg` not found after install**  
  - Make sure your environment's `bin`/`Scripts` directory is on `PATH`.
  - On many systems, this is handled automatically by virtual environments.

- **Very large `n` is slow**  
  - The implementation is CPU‑only and uses Python loops; for
    extremely large graphs, prefer the original CUDA code or a
    GPU‑accelerated Python reimplementation.


### 9. License and Citation

The original algorithm and CUDA implementation are due to:

> Sadegh Nobari, Xuesong Lu, Panagiotis Karras, Stéphane Bressan.  
> *Fast random graph generation.*  
> EDBT/ICDT 2011.

When using the Python port in academic work, please **cite the original
paper above** and mention that you used the Python reimplementation
from this repository.
