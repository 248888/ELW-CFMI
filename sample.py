"""Simulation data for the Section 4.1 MAR example in arXiv:2604.04567v1.

The main setting is the uniform example:

* d = 3
* X1 and X2 have uniform[0, 1] marginals with Pearson correlation 0.7
* X3 is independent uniform[0, 1]
* missingness patterns, using the paper's convention M_j = 1 means missing:
    m1 = (0, 0, 0), p1(x) = (x1 + x2) / 3
    m2 = (0, 1, 0), p2(x) = (2 - x1) / 3
    m3 = (1, 0, 0), p3(x) = (1 - x2) / 3

The rest of this code uses the project convention that ``observed_mask`` is
True for observed entries, so those three patterns become (1, 1, 1),
(1, 0, 1), and (0, 1, 1).
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm


PAPER_MISSING_PATTERNS = np.asarray(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ],
    dtype=np.int8,
)
OBSERVED_PATTERNS = (PAPER_MISSING_PATTERNS == 0)


@dataclass(frozen=True)
class SimulationDataset:
    x_full: np.ndarray
    x_missing: np.ndarray
    observed_mask: np.ndarray
    paper_missing_pattern: np.ndarray
    pattern_probabilities: np.ndarray
    x_test: np.ndarray
    true_quantile_x1: float
    distribution: str
    corr: float


def _latent_corr_for_uniform_pearson(target_corr: float) -> float:
    """Gaussian-copula latent correlation for target Pearson corr of uniforms."""

    if not -1.0 < target_corr < 1.0:
        raise ValueError("target_corr must be in (-1, 1).")
    return float(2.0 * np.sin(np.pi * target_corr / 6.0))


def sample_uniform_copula(
    n: int,
    *,
    corr: float = 0.7,
    seed: int | None = None,
    match_uniform_pearson: bool = True,
) -> np.ndarray:
    """Sample the Section 4.1 complete-data distribution."""

    rng = np.random.default_rng(seed)
    latent_corr = _latent_corr_for_uniform_pearson(corr) if match_uniform_pearson else corr
    cov = np.asarray(
        [
            [1.0, latent_corr, 0.0],
            [latent_corr, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    z = rng.multivariate_normal(mean=np.zeros(3), cov=cov, size=n)
    return ndtr(z).astype(np.float32)


def sample_gaussian(
    n: int,
    *,
    corr: float = 0.7,
    seed: int | None = None,
) -> np.ndarray:
    """Optional Appendix A.2 Gaussian variant."""

    rng = np.random.default_rng(seed)
    cov = np.asarray(
        [
            [1.0, corr, 0.0],
            [corr, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rng.multivariate_normal(mean=np.zeros(3), cov=cov, size=n).astype(np.float32)


def pattern_probabilities(
    x: np.ndarray,
    *,
    distribution: Literal["uniform", "gaussian"] = "uniform",
) -> np.ndarray:
    """Return probabilities for paper patterns m1, m2, m3."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3).")

    if distribution == "uniform":
        x1_score = x[:, 0]
        x2_score = x[:, 1]
    elif distribution == "gaussian":
        x1_score = ndtr(x[:, 0])
        x2_score = ndtr(x[:, 1])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    probs = np.column_stack(
        [
            (x1_score + x2_score) / 3.0,
            (2.0 - x1_score) / 3.0,
            (1.0 - x2_score) / 3.0,
        ]
    )
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float64)


def apply_section_4_1_missingness(
    x: np.ndarray,
    *,
    distribution: Literal["uniform", "gaussian"] = "uniform",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the non-monotone MAR mechanism from Section 4.1."""

    rng = np.random.default_rng(seed)
    probs = pattern_probabilities(x, distribution=distribution)
    uniforms = rng.random(x.shape[0])
    pattern_index = (uniforms[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    pattern_index = np.minimum(pattern_index, PAPER_MISSING_PATTERNS.shape[0] - 1)
    paper_patterns = PAPER_MISSING_PATTERNS[pattern_index]
    observed_mask = paper_patterns == 0
    x_missing = np.where(observed_mask, x, np.nan).astype(np.float32)
    return x_missing, observed_mask, paper_patterns, probs


def true_propensity_by_observed_pattern(
    pattern_probs: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Map observed-mask patterns to true P(M=m | X) arrays."""

    return {
        tuple(int(v) for v in observed_pattern): pattern_probs[:, j]
        for j, observed_pattern in enumerate(OBSERVED_PATTERNS)
    }


def make_simulation_dataset(
    *,
    n: int = 2000,
    test_n: int | None = None,
    corr: float = 0.7,
    seed: int = 0,
    distribution: Literal["uniform", "gaussian"] = "uniform",
) -> SimulationDataset:
    """Generate train-with-missingness and independent complete test data."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if test_n is None:
        test_n = n
    if test_n <= 0:
        raise ValueError("test_n must be positive.")

    if distribution == "uniform":
        sampler = sample_uniform_copula
        true_quantile = 0.1
    elif distribution == "gaussian":
        sampler = sample_gaussian
        true_quantile = float(norm.ppf(0.1))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    x_full = sampler(n, corr=corr, seed=seed)
    x_test = sampler(test_n, corr=corr, seed=seed + 1_000_003)
    x_missing, observed_mask, paper_patterns, probs = apply_section_4_1_missingness(
        x_full,
        distribution=distribution,
        seed=seed + 2_000_003,
    )

    return SimulationDataset(
        x_full=x_full,
        x_missing=x_missing,
        observed_mask=observed_mask,
        paper_missing_pattern=paper_patterns,
        pattern_probabilities=probs,
        x_test=x_test,
        true_quantile_x1=true_quantile,
        distribution=distribution,
        corr=corr,
    )


def save_simulation_dataset(dataset: SimulationDataset, path: str | Path) -> None:
    """Save a simulation dataset as a compressed NumPy archive."""

    np.savez_compressed(
        Path(path),
        x_full=dataset.x_full,
        x_missing=dataset.x_missing,
        observed_mask=dataset.observed_mask,
        paper_missing_pattern=dataset.paper_missing_pattern,
        pattern_probabilities=dataset.pattern_probabilities,
        x_test=dataset.x_test,
        true_quantile_x1=np.asarray(dataset.true_quantile_x1),
        distribution=np.asarray(dataset.distribution),
        corr=np.asarray(dataset.corr),
    )


def load_simulation_dataset(path: str | Path) -> SimulationDataset:
    """Load a dataset saved by ``save_simulation_dataset``."""

    with np.load(Path(path), allow_pickle=False) as data:
        return SimulationDataset(
            x_full=data["x_full"],
            x_missing=data["x_missing"],
            observed_mask=data["observed_mask"].astype(bool),
            paper_missing_pattern=data["paper_missing_pattern"],
            pattern_probabilities=data["pattern_probabilities"],
            x_test=data["x_test"],
            true_quantile_x1=float(data["true_quantile_x1"]),
            distribution=str(data["distribution"]),
            corr=float(data["corr"]),
        )


def missing_rate(observed_mask: np.ndarray) -> float:
    """Fraction of missing entries among all n * d entries."""

    return float(1.0 - np.asarray(observed_mask, dtype=bool).mean())


def main() -> None:
    parser = ArgumentParser(description="Generate the Section 4.1 simulation dataset.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--test-n", type=int, default=None)
    parser.add_argument("--corr", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distribution", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--output", type=Path, default=Path("section_4_1_simulation.npz"))
    args = parser.parse_args()

    dataset = make_simulation_dataset(
        n=args.n,
        test_n=args.test_n,
        corr=args.corr,
        seed=args.seed,
        distribution=args.distribution,
    )
    save_simulation_dataset(dataset, args.output)

    print(f"saved: {args.output}")
    print(f"x_full shape: {dataset.x_full.shape}")
    print(f"x_test shape: {dataset.x_test.shape}")
    print(f"missing rate: {missing_rate(dataset.observed_mask):.4f}")
    print(f"true 0.1 quantile of X1: {dataset.true_quantile_x1:.6f}")


if __name__ == "__main__":
    main()
