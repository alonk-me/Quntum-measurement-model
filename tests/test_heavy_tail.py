import numpy as np
import pytest

from quantum_measurement.utilities.heavy_tail import (
    bootstrap_hill_interval,
    hill_curve,
    hill_tail_index,
    running_max_share,
    running_mean_sem,
    survival_function,
)


def test_running_mean_sem_matches_direct_statistics():
    values = np.random.default_rng(1).normal(3.0, 2.0, size=50)
    counts, means, sems = running_mean_sem(values)
    assert counts[0] == 2 and counts[-1] == 50
    for index, count in enumerate(counts):
        prefix = values[:count]
        assert means[index] == pytest.approx(np.mean(prefix), rel=1e-12)
        assert sems[index] == pytest.approx(
            np.std(prefix, ddof=1) / np.sqrt(count), rel=1e-10
        )


def test_running_mean_sem_survives_one_huge_value():
    values = np.r_[np.random.default_rng(2).normal(size=1000), 1.0e12]
    _, _, sems = running_mean_sem(values)
    direct = np.std(values[:1000], ddof=1) / np.sqrt(1000)
    assert sems[-2] == pytest.approx(direct, rel=1e-10)


def test_gaussian_sem_scales_as_inverse_root_n_and_max_share_vanishes():
    values = np.random.default_rng(3).normal(0.0, 1.5, size=10_000)
    counts, _, sems = running_mean_sem(values)
    scaled = sems * np.sqrt(counts)
    np.testing.assert_allclose(scaled[[999, 4999, 9998]], 1.5, rtol=0.06)
    assert running_max_share(values)[-1] < 0.01


@pytest.mark.parametrize("alpha", [1.5, 3.0])
def test_hill_recovers_pareto_tail_index(alpha):
    rng = np.random.default_rng(4)
    values = rng.pareto(alpha, size=20_000) + 1.0
    estimate, lower, upper = bootstrap_hill_interval(
        values, k_fraction=0.05, n_boot=200, rng=np.random.default_rng(5)
    )
    assert lower <= alpha <= upper
    assert estimate == pytest.approx(alpha, rel=0.1)
    curve = hill_curve(values, [500, 1000])
    np.testing.assert_allclose(curve, alpha, rtol=0.12)


def test_cauchy_max_share_stays_order_one_fraction():
    values = np.random.default_rng(6).standard_cauchy(size=20_000)
    assert running_max_share(values)[-1] > 0.05


def test_survival_function_and_input_validation():
    magnitudes, survival = survival_function(np.array([-3.0, 1.0, 2.0, 0.5]))
    np.testing.assert_array_equal(magnitudes, [0.5, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(survival, [1.0, 0.75, 0.5, 0.25])
    with pytest.raises(ValueError):
        running_mean_sem(np.array([1.0]))
    with pytest.raises(ValueError):
        hill_tail_index(np.array([1.0, 2.0, 3.0]), 3)
    with pytest.raises(ValueError):
        hill_tail_index(np.array([0.0, 0.0, 1.0]), 1)
    with pytest.raises(ValueError):
        running_max_share(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        bootstrap_hill_interval(np.ones(10) + np.arange(10), 0.1, 10, rng=1)
