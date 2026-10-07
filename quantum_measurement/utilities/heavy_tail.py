"""Diagnostics for whether a sample mean behaves like a well-defined average.

These helpers support the tangent sample-scaling study.  For a finite-variance
sample the running mean settles, ``SEM * sqrt(N)`` is flat, the share of the
largest absolute value in the absolute sum tends to zero, and the tail index
of ``|X|`` exceeds two.  A tail index below two means infinite variance; below
one, an infinite mean.  None of these functions trims or reweights samples.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _one_dimensional(values: np.ndarray, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def running_mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(N, mean, sem)`` for ``N = 2..n`` in the given sample order.

    Welford's update is used so that large, rare values do not destroy the
    variance of the remaining samples through cancellation.  The standard
    error uses ``ddof=1``.
    """

    array = _one_dimensional(values)
    if array.size < 2:
        raise ValueError("values must contain at least two samples")
    means = np.empty(array.size)
    squared_deviation = np.empty(array.size)
    mean = 0.0
    accumulated = 0.0
    for index, value in enumerate(array):
        delta = value - mean
        mean += delta / (index + 1)
        accumulated += delta * (value - mean)
        means[index] = mean
        squared_deviation[index] = accumulated
    counts = np.arange(1, array.size + 1)
    sem = np.sqrt(squared_deviation[1:] / (counts[1:] - 1) / counts[1:])
    return counts[1:], means[1:], sem


def running_max_share(values: np.ndarray) -> np.ndarray:
    """Return the running ``max|x_i| / sum|x_i|`` for ``N = 1..n``."""

    magnitudes = np.abs(_one_dimensional(values))
    totals = np.cumsum(magnitudes)
    maxima = np.maximum.accumulate(magnitudes)
    return np.divide(
        maxima, totals, out=np.zeros_like(maxima), where=totals > 0.0
    )


def hill_tail_index(abs_values: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index from the ``k`` largest magnitudes.

    With descending order statistics ``x_(1) >= ... >= x_(n)``,
    ``alpha = k / sum_{i<=k} log(x_(i) / x_(k+1))``.
    """

    magnitudes = np.abs(_one_dimensional(abs_values, "abs_values"))
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    k = int(k)
    if not 1 <= k < magnitudes.size:
        raise ValueError("k must satisfy 1 <= k < number of samples")
    descending = np.sort(magnitudes)[::-1]
    threshold = descending[k]
    if threshold <= 0.0:
        raise ValueError("the (k+1)-th largest magnitude must be positive")
    log_excess = np.sum(np.log(descending[:k] / threshold))
    if log_excess <= 0.0:
        return float("inf")
    return float(k / log_excess)


def hill_curve(abs_values: np.ndarray, k_values: Iterable[int]) -> np.ndarray:
    """Return the Hill estimate for each ``k`` (a Hill plot)."""

    return np.array([hill_tail_index(abs_values, int(k)) for k in k_values])


def bootstrap_hill_interval(
    abs_values: np.ndarray,
    k_fraction: float,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return ``(alpha, lower, upper)`` with a percentile bootstrap interval.

    ``k = max(1, round(k_fraction * n))`` is used for the estimate and for
    every resample.  The generator must be supplied explicitly.
    """

    magnitudes = np.abs(_one_dimensional(abs_values, "abs_values"))
    if not 0.0 < k_fraction < 1.0:
        raise ValueError("k_fraction must lie strictly between zero and one")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    k = max(1, int(round(k_fraction * magnitudes.size)))
    estimate = hill_tail_index(magnitudes, k)
    samples = np.array(
        [
            hill_tail_index(
                rng.choice(magnitudes, size=magnitudes.size, replace=True), k
            )
            for _ in range(n_boot)
        ]
    )
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return estimate, float(lower), float(upper)


def survival_function(abs_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted magnitudes and the empirical ``P(|X| >= x)`` at each."""

    magnitudes = np.sort(np.abs(_one_dimensional(abs_values, "abs_values")))
    n = magnitudes.size
    return magnitudes, (n - np.arange(n)) / n
