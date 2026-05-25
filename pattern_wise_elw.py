"""Pattern-wise empirical likelihood weights for ELW-CFMI.

The main entry point is ``pattern_wise_elw_weights``.  It implements the
algorithm in the prompt when estimated pattern propensities are already
available.  The helper ``estimate_propensity_by_pattern`` gives a practical
baseline for producing those propensities with one-vs-rest classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional, Tuple

import numpy as np


Pattern = Hashable


@dataclass(frozen=True)
class PatternELWResult:
    """ELW result for one missingness pattern."""

    pattern: Pattern
    indices: np.ndarray
    propensity: np.ndarray
    weights: np.ndarray
    raw_weights: np.ndarray
    p_hat: float
    alpha: float
    lambda_: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class PatternWiseELWResult:
    """ELW result for all observed patterns.

    ``within_pattern_weights`` sums to one inside each pattern.
    ``training_weights`` multiplies those weights by ``rho_m``; with the
    default ``rho_m = n_m / n``, it sums to one over the full sample.
    """

    within_pattern_weights: np.ndarray
    training_weights: np.ndarray
    pattern_results: Dict[Pattern, PatternELWResult]


def observed_mask_from_nan(x: np.ndarray) -> np.ndarray:
    """Return a 1/0 observed mask where finite/non-nan entries are observed."""

    x = np.asarray(x)
    return ~np.isnan(x)


def pattern_labels_from_mask(mask: np.ndarray) -> np.ndarray:
    """Convert an ``n x d`` observed mask into hashable tuple pattern labels."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    labels = np.empty(mask.shape[0], dtype=object)
    labels[:] = [tuple(row.astype(int).tolist()) for row in mask]
    return labels


def unique_in_order(values: Iterable[Pattern]) -> Tuple[Pattern, ...]:
    """Return unique labels while preserving first-seen order."""

    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return tuple(out)


def _pattern_equal(labels: np.ndarray, pattern: Pattern) -> np.ndarray:
    """Elementwise equality for scalar labels and tuple-valued mask labels."""

    return np.fromiter((label == pattern for label in labels), dtype=bool, count=len(labels))


def _normalize_patterns(patterns: np.ndarray) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=object)
    if patterns.ndim == 1:
        labels = []
        for value in patterns:
            if isinstance(value, np.ndarray):
                labels.append(tuple(value.astype(int).tolist()))
            elif isinstance(value, (list, tuple)):
                labels.append(tuple(np.asarray(value).astype(int).tolist()))
            else:
                labels.append(value)
        out = np.empty(len(labels), dtype=object)
        out[:] = labels
        return out
    if patterns.ndim == 2:
        return pattern_labels_from_mask(patterns)
    raise ValueError("patterns must be a one-dimensional label array or a two-dimensional mask")


def _as_propensity_for_pattern(
    propensity: np.ndarray,
    pattern_indices: np.ndarray,
    n_total: int,
) -> np.ndarray:
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    n_pattern = pattern_indices.size
    if propensity.size == n_total:
        return propensity[pattern_indices]
    if propensity.size == n_pattern:
        return propensity
    raise ValueError(
        "Each propensity array must have length n or length n_m for its pattern; "
        f"got {propensity.size}, expected {n_total} or {n_pattern}."
    )


def elw_weights_from_propensity(
    propensity: np.ndarray,
    n_total: int,
    *,
    clip: Optional[Tuple[float, float]] = (1e-6, 1.0 - 1e-6),
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, bool, int]:
    """Compute ELW weights for a single pattern.

    Parameters
    ----------
    propensity:
        Estimated ``pi_m(y_m,j)`` for the observations in pattern ``m``.
    n_total:
        Full sample size ``n``.
    clip:
        Optional lower/upper clipping for numerical stability.  Use ``None``
        if you want exact inputs with no clipping.

    Returns
    -------
    weights, raw_weights, p_hat, alpha, lambda_, converged, iterations
    """

    pi = np.asarray(propensity, dtype=float).reshape(-1)
    if pi.size == 0:
        raise ValueError("propensity must contain at least one observation")
    if not np.all(np.isfinite(pi)):
        raise ValueError("propensity contains nan or infinite values")
    if n_total < pi.size:
        raise ValueError("n_total must be at least the pattern sample size")

    if clip is not None:
        low, high = clip
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("clip must satisfy 0 <= low < high <= 1")
        pi = np.clip(pi, low, high)
    elif np.any((pi < 0.0) | (pi > 1.0)):
        raise ValueError("propensity must be in [0, 1] when clip is None")

    n_pattern = pi.size
    p_hat = n_pattern / float(n_total)
    xi = p_hat + (1.0 - p_hat) * pi

    lower = float(np.min(pi))
    upper = float(np.min(xi))

    def score(alpha: float) -> float:
        denom = xi - alpha
        if np.any(denom <= 0.0):
            return -np.inf
        return float(np.sum((pi - alpha) / denom))

    f_lower = score(lower)
    if abs(f_lower) <= tol:
        alpha = lower
        converged = True
        iterations = 0
    else:
        # Stay just inside the interval to avoid the pole at min(xi).
        left = lower
        right = np.nextafter(upper, lower)
        f_right = score(right)

        if not (f_lower > 0.0 and f_right < 0.0):
            raise RuntimeError(
                "Bisection bracket failed. Check whether the propensity scores "
                "are valid and have enough variation."
            )

        alpha = 0.5 * (left + right)
        converged = False
        iterations = max_iter
        for iteration in range(1, max_iter + 1):
            alpha = 0.5 * (left + right)
            f_mid = score(alpha)
            if abs(f_mid) <= tol or (right - left) <= tol * max(1.0, abs(alpha)):
                converged = True
                iterations = iteration
                break
            if f_mid > 0.0:
                left = alpha
            else:
                right = alpha

    if abs(1.0 - alpha) <= np.finfo(float).eps:
        raise RuntimeError("alpha is too close to one; lambda is unstable")

    lambda_ = (n_total - n_pattern) / (n_pattern * (1.0 - alpha))
    raw_weights = (1.0 / n_pattern) / (1.0 + lambda_ * (pi - alpha))
    if np.any(raw_weights <= 0.0) or not np.all(np.isfinite(raw_weights)):
        raise RuntimeError("computed invalid ELW weights")

    weights = raw_weights / np.sum(raw_weights)
    return weights, raw_weights, p_hat, alpha, lambda_, converged, iterations


def pattern_wise_elw_weights(
    patterns: np.ndarray,
    propensity_by_pattern: Mapping[Pattern, np.ndarray],
    *,
    rho_by_pattern: Optional[Mapping[Pattern, float]] = None,
    clip: Optional[Tuple[float, float]] = (1e-6, 1.0 - 1e-6),
    tol: float = 1e-12,
    max_iter: int = 200,
) -> PatternWiseELWResult:
    """Compute pattern-wise ELW weights for all supplied patterns.

    Parameters
    ----------
    patterns:
        Either an ``n``-length pattern label array or an ``n x d`` observed mask.
    propensity_by_pattern:
        Dictionary keyed by pattern.  Each value can be either length ``n`` or
        length ``n_m``.  If length ``n``, the function selects rows in that
        pattern before applying the algorithm.
    rho_by_pattern:
        Optional pattern-level weights.  Defaults to ``n_m / n``.
    """

    labels = _normalize_patterns(patterns)
    n_total = labels.size
    within_pattern_weights = np.zeros(n_total, dtype=float)
    training_weights = np.zeros(n_total, dtype=float)
    results: Dict[Pattern, PatternELWResult] = {}

    for pattern in unique_in_order(labels):
        if pattern not in propensity_by_pattern:
            raise KeyError(f"Missing propensity scores for pattern {pattern!r}")

        indices = np.flatnonzero(_pattern_equal(labels, pattern))
        pi = _as_propensity_for_pattern(propensity_by_pattern[pattern], indices, n_total)
        weights, raw_weights, p_hat, alpha, lambda_, converged, iterations = (
            elw_weights_from_propensity(
                pi,
                n_total,
                clip=clip,
                tol=tol,
                max_iter=max_iter,
            )
        )

        rho = p_hat if rho_by_pattern is None else float(rho_by_pattern[pattern])
        within_pattern_weights[indices] = weights
        training_weights[indices] = rho * weights
        results[pattern] = PatternELWResult(
            pattern=pattern,
            indices=indices,
            propensity=pi,
            weights=weights,
            raw_weights=raw_weights,
            p_hat=p_hat,
            alpha=alpha,
            lambda_=lambda_,
            converged=converged,
            iterations=iterations,
        )

    return PatternWiseELWResult(
        within_pattern_weights=within_pattern_weights,
        training_weights=training_weights,
        pattern_results=results,
    )


def default_logistic_estimator(random_state: int = 0):
    """Create a simple probability model for pattern propensities.

    This import is lazy so the ELW functions can be used without scikit-learn.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )


def estimate_propensity_by_pattern(
    x: np.ndarray,
    observed_mask: Optional[np.ndarray] = None,
    *,
    patterns: Optional[np.ndarray] = None,
    estimator_factory: Optional[Callable[[], object]] = None,
    cross_fit: int = 5,
    clip: Tuple[float, float] = (1e-3, 1.0 - 1e-3),
    random_state: int = 0,
) -> Dict[Pattern, np.ndarray]:
    """Estimate ``pi_m(y) = P(M=m | X observed under pattern m = y)``.

    For each observed pattern ``m``, this function fits a one-vs-rest binary
    classifier using the coordinates observed in pattern ``m``.  Rows that do
    not observe all those coordinates are excluded from that pattern-specific
    classifier.  The returned arrays are only for rows that actually belong to
    each pattern, so they can be passed directly to ``pattern_wise_elw_weights``.

    Notes
    -----
    This is a baseline implementation.  For serious experiments, prefer
    cross-fitting plus a well-calibrated model chosen by validation.
    """

    try:
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "estimate_propensity_by_pattern requires scikit-learn. "
            "Install scikit-learn or pass externally estimated propensities to "
            "pattern_wise_elw_weights."
        ) from exc

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array")

    if observed_mask is None:
        observed_mask = observed_mask_from_nan(x)
    else:
        observed_mask = np.asarray(observed_mask).astype(bool)
    if observed_mask.shape != x.shape:
        raise ValueError("observed_mask must have the same shape as x")

    labels = pattern_labels_from_mask(observed_mask) if patterns is None else _normalize_patterns(patterns)
    if labels.size != x.shape[0]:
        raise ValueError("patterns must have one label per row")

    if estimator_factory is None:
        estimator_factory = lambda: default_logistic_estimator(random_state=random_state)

    low, high = clip
    propensities: Dict[Pattern, np.ndarray] = {}
    rng_seed = random_state

    for pattern in unique_in_order(labels):
        pattern_index = np.flatnonzero(_pattern_equal(labels, pattern))
        pattern_array = np.asarray(pattern if isinstance(pattern, tuple) else (), dtype=int)

        if pattern_array.size == x.shape[1]:
            feature_cols = np.flatnonzero(pattern_array.astype(bool))
        else:
            # For non-mask labels, use all columns that are observed in every
            # row of the target pattern.
            feature_cols = np.flatnonzero(np.all(observed_mask[pattern_index], axis=0))

        if feature_cols.size == 0:
            eligible = np.ones(x.shape[0], dtype=bool)
            x_fit = np.empty((x.shape[0], 0), dtype=float)
        else:
            eligible = np.all(observed_mask[:, feature_cols], axis=1)
            x_fit = x[eligible][:, feature_cols]

        eligible_labels = labels[eligible]
        y_fit = _pattern_equal(eligible_labels, pattern).astype(int)
        prevalence = float(np.mean(y_fit))

        if feature_cols.size == 0 or np.unique(y_fit).size < 2:
            pi_target = np.full(pattern_index.size, prevalence, dtype=float)
            propensities[pattern] = np.clip(pi_target, low, high)
            continue

        min_class_count = int(np.min(np.bincount(y_fit)))
        n_splits = min(int(cross_fit), min_class_count)
        predictions = np.full(y_fit.shape[0], np.nan, dtype=float)

        if n_splits >= 2:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=rng_seed,
            )
            base_estimator = estimator_factory()
            for train_idx, test_idx in splitter.split(x_fit, y_fit):
                estimator = clone(base_estimator)
                estimator.fit(x_fit[train_idx], y_fit[train_idx])
                predictions[test_idx] = estimator.predict_proba(x_fit[test_idx])[:, 1]
        else:
            estimator = estimator_factory()
            estimator.fit(x_fit, y_fit)
            predictions[:] = estimator.predict_proba(x_fit)[:, 1]

        if np.any(~np.isfinite(predictions)):
            estimator = estimator_factory()
            estimator.fit(x_fit, y_fit)
            missing = ~np.isfinite(predictions)
            predictions[missing] = estimator.predict_proba(x_fit[missing])[:, 1]

        eligible_index = np.flatnonzero(eligible)
        target_position = np.searchsorted(eligible_index, pattern_index)
        if not np.array_equal(eligible_index[target_position], pattern_index):
            raise RuntimeError("Target pattern rows are not eligible for their own features")

        pi_target = predictions[target_position]
        propensities[pattern] = np.clip(pi_target, low, high)

    return propensities


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    x_demo = rng.normal(size=(80, 4))
    observed = rng.random(size=x_demo.shape) > 0.25
    x_demo[~observed] = np.nan

    patterns_demo = pattern_labels_from_mask(observed)
    propensity_demo = estimate_propensity_by_pattern(
        x_demo,
        observed_mask=observed,
        patterns=patterns_demo,
        cross_fit=3,
    )
    result_demo = pattern_wise_elw_weights(patterns_demo, propensity_demo)

    print("Number of patterns:", len(result_demo.pattern_results))
    print("Sum of training weights:", result_demo.training_weights.sum())
    print("First 10 training weights:", result_demo.training_weights[:10])
