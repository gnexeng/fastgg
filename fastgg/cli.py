"""
Command-line interface for the Python FASTRGG implementation.

This script is intended to be run as:

    python -m fastgg.cli -n 1000 -p 0.1 -r 3 -a per
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from .algorithms import Algorithm, RunResult, generate_graph


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fastgg-python",
        description="Python implementation of the FASTRGG random graph generator (G(n, p)).",
    )

    parser.add_argument(
        "-n",
        "--vertices",
        type=int,
        required=True,
        help="Number of vertices (n).",
    )
    parser.add_argument(
        "-p",
        "--probability",
        type=float,
        required=True,
        help="Edge probability p in [0, 1].",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of runs to perform (default: 1).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=7,
        help="Base random seed (default: 7). Each run uses seed + run_index.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=[a.value for a in Algorithm],
        default=Algorithm.PER.value,
        help="Algorithm variant: per, pzer, or pprezer (default: per).",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to a CSV file where per-run statistics will be appended. "
            "If omitted, results are only printed to stdout."
        ),
    )
    parser.add_argument(
        "--edges-out",
        type=Path,
        default=None,
        help=(
            "Optional path to a CSV file where the edges from the LAST run will be written "
            "as rows of 'u,v' pairs (0-based vertex indices)."
        ),
    )

    return parser.parse_args(argv)


def _append_log_csv(path: Path, results: List[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "run_index",
                    "n",
                    "p",
                    "algorithm",
                    "seed",
                    "elapsed_ms",
                    "edge_count",
                ]
            )
        for idx, res in enumerate(results):
            writer.writerow(
                [
                    idx,
                    res.n,
                    res.p,
                    res.algorithm.value,
                    res.seed,
                    f"{res.elapsed_ms:.3f}",
                    res.edge_count,
                ]
            )


def _write_edges_csv(path: Path, res: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v"])
        for u, v in res.as_uv_pairs():
            writer.writerow([u, v])


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)

    algorithm = Algorithm(ns.algorithm)
    n = ns.vertices
    p = ns.probability
    base_seed = ns.seed
    runs = max(1, ns.runs)

    results: List[RunResult] = []

    print(
        f"Generating G(n={n}, p={p}) with algorithm={algorithm.value}, "
        f"runs={runs}, base_seed={base_seed}"
    )

    for i in range(runs):
        seed = base_seed + i
        res = generate_graph(n=n, p=p, algorithm=algorithm, seed=seed)
        results.append(res)
        print(
            f"Run {i}: seed={seed}, edges={res.edge_count}, "
            f"elapsed={res.elapsed_ms:.3f} ms"
        )

    if ns.log_csv is not None:
        _append_log_csv(ns.log_csv, results)
        print(f"Wrote per-run statistics to '{ns.log_csv}'.")

    if ns.edges_out is not None and results:
        _write_edges_csv(ns.edges_out, results[-1])
        print(
            f"Wrote edges for last run (run {runs-1}) to '{ns.edges_out}' "
            "as CSV (columns: u,v)."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
