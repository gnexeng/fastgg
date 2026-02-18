"""
Core Python implementations of the FASTRGG algorithms.

These implementations are designed to be:
- **Correct in distribution** for G(n, p) Erdős–Rényi graphs
- **Simple and portable**, using only the Python standard library

They do **not** attempt to reproduce the exact CUDA execution structure,
but they preserve the high-level algorithmic ideas:
- PER: test each potential edge independently with probability ``p``
- PZER: use geometric skips to jump between present edges
- PPreZER: like PZER but with a small precomputed table for short skips
"""

from __future__ import annotations

import enum
import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


M: int = 2_147_483_647  # same modulus as the original LCG
A: int = 16_807  # multiplier in the original CUDA code


class Algorithm(str, enum.Enum):
    PER = "per"
    PZER = "pzer"
    PPREZER = "pprezer"


@dataclass
class RunResult:
    """Result for a single run of a generator algorithm."""

    n: int
    p: float
    algorithm: Algorithm
    seed: int
    elapsed_ms: float
    edges_linear: List[int]

    @property
    def edge_count(self) -> int:
        return len(self.edges_linear)

    def as_uv_pairs(self) -> List[Tuple[int, int]]:
        """Return edges as (u, v) pairs for 0 ≤ u, v < n."""
        n = self.n
        return [(idx // n, idx % n) for idx in self.edges_linear]


def _lcg_sequence(seed: int) -> Iterable[int]:
    """Simple LCG sequence matching the constants in the CUDA version."""
    state = seed % M
    while True:
        state = (state * A) % M
        yield state


def _validate_params(n: int, p: float) -> None:
    if n <= 0:
        raise ValueError("n (number of vertices) must be positive.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p (probability) must be in [0, 1].")


def generate_graph(
    n: int,
    p: float,
    algorithm: Algorithm = Algorithm.PER,
    seed: int = 7,
) -> RunResult:
    """
    Generate a random Erdős–Rényi graph G(n, p).

    Edges are over the full n×n directed adjacency matrix (including self-loops),
    which matches the effective behavior of the original CUDA implementation
    where ``TotalE = v^2``.

    Parameters
    ----------
    n:
        Number of vertices.
    p:
        Edge probability (0 ≤ p ≤ 1).
    algorithm:
        Which algorithm variant to use: PER, PZER, or PPreZER.
    seed:
        Integer random seed. Different algorithms use this slightly differently,
        but using the same seed should give comparable densities.
    """
    _validate_params(n, p)

    start = time.perf_counter()

    if algorithm is Algorithm.PER:
        edges = _run_per(n, p, seed)
    elif algorithm is Algorithm.PZER:
        edges = _run_pzer(n, p, seed)
    elif algorithm is Algorithm.PPREZER:
        edges = _run_pprezer(n, p, seed, precompute=10)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    elapsed_ms = (time.perf_counter() - start) * 1_000.0

    return RunResult(
        n=n,
        p=p,
        algorithm=algorithm,
        seed=seed,
        elapsed_ms=elapsed_ms,
        edges_linear=edges,
    )


def _run_per(n: int, p: float, seed: int) -> List[int]:
    """
    PER: test each of the n^2 potential edges independently with probability p.

    This is the most direct algorithm and closely mirrors the original idea:
    each potential edge gets an independent Bernoulli(p) trial.
    """
    total_e = n * n
    threshold = int(p * M)
    gen = _lcg_sequence(seed)

    edges: List[int] = []
    for idx in range(total_e):
        if next(gen) < threshold:
            edges.append(idx)
    return edges


def _geometric_skip(p: float, rng: random.Random) -> int:
    """
    Sample a geometric( p ) skip length (support {1, 2, ...}).

    This uses the standard inverse-CDF method:
        skip = floor( log(1-U) / log(1-p) ) + 1
    where U ~ Uniform(0, 1).
    """
    if p <= 0.0:
        # No edges at all
        return math.inf  # type: ignore[return-value]
    if p >= 1.0:
        # Every position is an edge
        return 1

    u = rng.random()
    return math.floor(math.log(1.0 - u) / math.log(1.0 - p)) + 1


def _run_pzer(n: int, p: float, seed: int) -> List[int]:
    """
    PZER: use geometric skips between edges across the n^2 positions.

    This is a sequential, CPU-friendly specialization of the skip-based idea
    from the CUDA implementation. It produces a G(n, p) distribution but
    may be faster than PER when the expected number of edges is small.
    """
    total_e = n * n
    rng = random.Random(seed)

    edges: List[int] = []
    pos = 0
    while pos < total_e:
        skip = _geometric_skip(p, rng)
        if not math.isfinite(skip):
            break
        pos += int(skip)
        if pos <= total_e:
            edges.append(pos - 1)
    return edges


def _precompute_thresholds(p: float, k: int) -> Sequence[float]:
    """
    Precompute cumulative probabilities for the first k geometric skip values.

    P(X = i) = (1-p)^(i-1) * p, for i ≥ 1
    """
    thresholds: List[float] = []
    cumulative = 0.0
    for i in range(1, k + 1):
        prob_i = (1.0 - p) ** (i - 1) * p
        cumulative += prob_i
        thresholds.append(cumulative)
    return thresholds


def _run_pprezer(n: int, p: float, seed: int, precompute: int = 10) -> List[int]:
    """
    PPreZER: geometric-skip method with a small precomputed skip table.

    For the first ``precompute`` skip lengths we avoid logarithms and instead
    use a simple table lookup. For larger skips, we fall back to the general
    geometric sampler.
    """
    total_e = n * n
    rng = random.Random(seed)

    if p <= 0.0:
        return []
    if p >= 1.0:
        # Full dense graph
        return list(range(total_e))

    thresholds = _precompute_thresholds(p, precompute)

    def sample_skip() -> int:
        u = rng.random()
        # Table lookup for short skips
        for i, t in enumerate(thresholds, start=1):
            if u <= t:
                return i
        # Tail: generic geometric
        return _geometric_skip(p, rng)

    edges: List[int] = []
    pos = 0
    while pos < total_e:
        skip = sample_skip()
        pos += skip
        if pos <= total_e:
            edges.append(pos - 1)
    return edges
