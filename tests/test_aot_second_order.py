from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quantum_measurement.aot_single_qubit import (
    AoTSecondOrderHistory,
    _measurement_operators_second_order,
    _premeasurement_observables_second_order,
    dressel_no_drive_reference,
    exact_unitary,
    rademacher_noise,
    second_order_summary,
    simulate_aot_second_order_cpu,
    simulate_ensemble,
    validate_aot_second_order_finite_difference,
)


def test_second_order_result_contract_and_seeded_reproducibility():
    arguments = dict(
        gamma=2.0,
        J=1.0,
        dt=0.005,
        n_burnin=10,
        n_samples=40,
        n_trajectories=3,
        seed=73,
    )
    first = simulate_aot_second_order_cpu(**arguments)
    second = simulate_aot_second_order_cpu(**arguments)

    for field_name in (
        "q",
        "dq_dtheta",
        "d2q_dtheta2",
        "Q",
        "dQ_dtheta",
        "d2Q_dtheta2",
        "final_z",
        "max_first_tangent_norm",
        "max_second_tangent_norm",
        "max_norm_error",
        "max_first_constraint_error",
        "max_second_constraint_error",
    ):
        first_values = getattr(first, field_name)
        np.testing.assert_array_equal(first_values, getattr(second, field_name))
        assert first_values.shape == (3,)
        assert not first_values.flags.writeable

    assert first.n_trajectories == 3
    assert first.n_samples == 40
    assert first.averaging_time == 0.2
    assert first.noise_kind == "rademacher"
    assert first.noise_shape == (3, 50)
    assert len(first.noise_hash) == 64
    assert first.noise_hash == second.noise_hash
    assert first.seed == 73
    assert first.derivative_parameter == "log_gamma_at_fixed_J"
    np.testing.assert_allclose(first.log_g, np.log(0.5))
    summary = second_order_summary(first)
    np.testing.assert_allclose(summary["q_mean"], np.mean(first.q))
    np.testing.assert_allclose(
        summary["d2q_dtheta2_sem"],
        np.std(first.d2q_dtheta2, ddof=1) / np.sqrt(first.n_trajectories),
    )
    with pytest.raises(FrozenInstanceError):
        first.gamma = 3.0
    with pytest.raises(ValueError, match=r"undefined when J=0"):
        simulate_aot_second_order_cpu(
            gamma=2.0,
            J=0.0,
            dt=0.005,
            n_burnin=0,
            n_samples=2,
            seed=1,
        ).log_g


def test_second_order_noise_contract_and_input_validation():
    base = dict(gamma=1.0, J=0.0, dt=0.01, n_burnin=2, n_samples=3)
    explicit = np.array([1, -1, 1, 1, -1], dtype=np.int8)

    result = simulate_aot_second_order_cpu(**base, noise=explicit)
    repeated = simulate_aot_second_order_cpu(**base, noise=explicit.copy())
    assert result.noise_shape == (1, 5)
    assert result.seed is None
    for field_name in ("q", "dq_dtheta", "d2q_dtheta2", "final_z"):
        np.testing.assert_array_equal(
            getattr(result, field_name), getattr(repeated, field_name)
        )

    with pytest.raises(ValueError, match="exactly one"):
        simulate_aot_second_order_cpu(**base)
    with pytest.raises(ValueError, match="exactly one"):
        simulate_aot_second_order_cpu(**base, noise=explicit, seed=2)
    with pytest.raises(ValueError, match="exactly 5 steps"):
        simulate_aot_second_order_cpu(**base, noise=explicit[:-1])
    with pytest.raises(ValueError, match=r"-1 or \+1"):
        simulate_aot_second_order_cpu(**base, noise=np.array([1, 0, 1, 1, 1]))

    invalid_cases = (
        (dict(gamma=0.0), "gamma"),
        (dict(J=-1.0), "J"),
        (dict(dt=0.0), "dt"),
        (dict(n_burnin=-1), "n_burnin"),
        (dict(n_samples=0), "n_samples"),
        (dict(n_trajectories=0), "n_trajectories"),
        (dict(psi0=np.zeros(2)), "psi0"),
        (dict(psi0=np.ones(3)), "psi0"),
    )
    for override, message in invalid_cases:
        arguments = dict(base, seed=1)
        arguments.update(override)
        with pytest.raises(ValueError, match=message):
            simulate_aot_second_order_cpu(**arguments)


def test_premeasurement_observables_match_direct_matrix_expressions():
    psi = np.array(
        [[1.0, 2.0], [1.0 + 1.0j, -2.0 + 0.5j]], dtype=complex
    )
    psi /= np.linalg.norm(psi, axis=1)[:, np.newaxis]
    eta = np.array(
        [[0.2 + 0.1j, -0.3j], [-0.1 + 0.4j, 0.2 - 0.2j]], dtype=complex
    )
    phi = np.array(
        [[-0.3 + 0.2j, 0.1], [0.25j, -0.4 - 0.1j]], dtype=complex
    )
    sigma_z = np.diag([1.0, -1.0])

    z, u, v = _premeasurement_observables_second_order(psi, eta, phi)
    expected_z = np.array([np.vdot(state, sigma_z @ state).real for state in psi])
    expected_u = np.array(
        [2.0 * np.vdot(tangent, sigma_z @ state).real for state, tangent in zip(psi, eta)]
    )
    expected_v = np.array(
        [
            2.0 * np.vdot(second, sigma_z @ state).real
            + 2.0 * np.vdot(first, sigma_z @ first).real
            for state, first, second in zip(psi, eta, phi)
        ]
    )
    np.testing.assert_allclose(z, expected_z, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(u, expected_u, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(v, expected_v, rtol=0.0, atol=2.0e-16)


@pytest.mark.parametrize("xi_value", [-1.0, 1.0])
def test_measurement_operator_derivatives_have_central_difference_convergence(
    xi_value,
):
    epsilon = 0.13
    z0 = np.array([0.2, -0.35])
    u0 = np.array([0.17, -0.11])
    v0 = np.array([-0.08, 0.09])
    xi = np.full(2, xi_value)
    _, first, second = _measurement_operators_second_order(
        z0, u0, v0, xi, epsilon
    )

    def operator(theta):
        z = z0 + u0 * theta + 0.5 * v0 * theta * theta
        zeros = np.zeros_like(z)
        return _measurement_operators_second_order(
            z, zeros, zeros, xi, epsilon * np.exp(0.5 * theta)
        )[0]

    errors = []
    for h in (4.0e-2, 2.0e-2, 1.0e-2):
        plus = operator(h)
        central = operator(0.0)
        minus = operator(-h)
        first_fd = (plus - minus) / (2.0 * h)
        second_fd = (plus - 2.0 * central + minus) / (h * h)
        errors.append(
            (
                np.max(np.abs(first - first_fd)),
                np.max(np.abs(second - second_fd)),
            )
        )
    errors = np.asarray(errors)
    assert errors[0, 0] / errors[1, 0] > 3.8
    assert errors[1, 0] / errors[2, 0] > 3.8
    assert errors[0, 1] / errors[1, 1] > 3.5
    assert errors[1, 1] / errors[2, 1] > 3.0


def test_pathwise_and_aggregate_second_derivatives_match_same_noise():
    noise = rademacher_noise(n_trajectories=4, n_steps=180, seed=123)
    validation = validate_aot_second_order_finite_difference(
        gamma=2.8,
        J=1.0,
        dt=0.005,
        n_burnin=30,
        n_samples=150,
        h=0.0025,
        noise=noise,
    )

    np.testing.assert_allclose(
        validation.history.u,
        validation.finite_difference_u,
        rtol=0.0,
        atol=1.5e-6,
    )
    np.testing.assert_allclose(
        validation.history.v,
        validation.finite_difference_v,
        rtol=0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        validation.tangent.dq_dtheta,
        validation.finite_difference_dq_dtheta,
        rtol=0.0,
        atol=1.5e-7,
    )
    np.testing.assert_allclose(
        validation.tangent.d2q_dtheta2,
        validation.finite_difference_d2q_dtheta2,
        rtol=0.0,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        validation.tangent.dQ_dtheta,
        validation.finite_difference_dQ_dtheta,
        rtol=0.0,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        validation.tangent.d2Q_dtheta2,
        validation.finite_difference_d2Q_dtheta2,
        rtol=0.0,
        atol=3.0e-6,
    )

    result = validation.tangent
    assert validation.history.z.shape == (4, 180)
    assert result.n_samples == 150
    # The first retained tangent is already nonzero.  A burn-in reset would
    # make this negative-control assertion fail even if later samples evolved.
    assert np.max(np.abs(validation.history.u[:, 30])) > 0.1
    assert np.max(result.max_norm_error) < 2.0e-14
    assert np.max(result.max_first_constraint_error) < 2.0e-14
    assert np.max(result.max_second_constraint_error) < 2.0e-14


@pytest.mark.parametrize(
    "gamma,n_steps,n_burnin,h",
    [(0.8, 60, 10, 3.0e-3), (4.0, 250, 40, 1.0e-3), (16.0, 600, 100, 2.0e-4)],
)
def test_pathwise_derivatives_cover_strengths_and_path_lengths(
    gamma, n_steps, n_burnin, h
):
    noise = rademacher_noise(2, n_steps, seed=17 + n_steps)
    validation = validate_aot_second_order_finite_difference(
        gamma=gamma,
        J=1.0,
        dt=0.005,
        n_burnin=n_burnin,
        n_samples=n_steps - n_burnin,
        h=h,
        noise=noise,
    )
    np.testing.assert_allclose(
        validation.history.u,
        validation.finite_difference_u,
        rtol=0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        validation.history.v,
        validation.finite_difference_v,
        rtol=0.0,
        atol=5.0e-6,
    )


def test_log_step_sweep_shows_truncation_convergence_and_roundoff_floor():
    noise = rademacher_noise(3, 120, seed=667)
    errors = []
    for h in (0.04, 0.02, 0.01, 0.005):
        validation = validate_aot_second_order_finite_difference(
            gamma=2.8,
            J=1.0,
            dt=0.005,
            n_burnin=20,
            n_samples=100,
            h=h,
            noise=noise,
        )
        errors.append(
            np.sqrt(
                np.mean(
                    (
                        validation.tangent.d2q_dtheta2
                        - validation.finite_difference_d2q_dtheta2
                    )
                    ** 2
                )
            )
        )
    errors = np.asarray(errors)
    np.testing.assert_array_less(3.7, errors[:-1] / errors[1:])

    cancellation = validate_aot_second_order_finite_difference(
        gamma=2.8,
        J=1.0,
        dt=0.005,
        n_burnin=20,
        n_samples=100,
        h=1.0e-6,
        noise=noise,
    )
    cancellation_error = np.sqrt(
        np.mean(
            (
                cancellation.tangent.d2q_dtheta2
                - cancellation.finite_difference_d2q_dtheta2
            )
            ** 2
        )
    )
    assert cancellation_error > 100.0 * errors[-1]


def test_exact_unitary_preserves_both_tangent_constraints():
    psi = np.array([1.0 + 0.3j, -0.2 + 0.7j], dtype=complex)
    psi /= np.linalg.norm(psi)
    eta = np.array([0.4 - 0.1j, 0.2 + 0.3j], dtype=complex)
    eta -= psi * np.vdot(psi, eta).real
    phi = np.array([-0.1 + 0.5j, 0.6 - 0.2j], dtype=complex)
    second_residual = np.vdot(psi, phi).real + np.vdot(eta, eta).real
    phi -= second_residual * psi

    unitary = exact_unitary(J=1.7, dt=0.013)
    propagated = [unitary @ vector for vector in (psi, eta, phi)]
    psi_after, eta_after, phi_after = propagated

    np.testing.assert_allclose(np.vdot(psi_after, psi_after), 1.0, atol=3.0e-16)
    np.testing.assert_allclose(np.vdot(psi_after, eta_after).real, 0.0, atol=2.0e-16)
    np.testing.assert_allclose(
        np.vdot(psi_after, phi_after).real + np.vdot(eta_after, eta_after).real,
        0.0,
        atol=3.0e-16,
    )


def test_extensive_fields_include_all_explicit_gamma_derivatives():
    result = simulate_aot_second_order_cpu(
        gamma=4.0,
        J=1.0,
        dt=0.005,
        n_burnin=20,
        n_samples=100,
        n_trajectories=3,
        seed=8,
    )
    prefactor = result.gamma * result.averaging_time
    np.testing.assert_allclose(result.Q, prefactor * result.q)
    np.testing.assert_allclose(
        result.dQ_dtheta, prefactor * (result.q + result.dq_dtheta)
    )
    np.testing.assert_allclose(
        result.d2Q_dtheta2,
        prefactor
        * (result.q + 2.0 * result.dq_dtheta + result.d2q_dtheta2),
    )


def test_new_path_preserves_existing_first_order_behavior():
    noise = rademacher_noise(n_trajectories=6, n_steps=400, seed=72)
    old, _ = simulate_ensemble(4.0, noise, dt=0.005, burn_in=80)
    new = simulate_aot_second_order_cpu(
        gamma=4.0,
        J=1.0,
        dt=0.005,
        n_burnin=80,
        n_samples=320,
        noise=noise,
    )

    np.testing.assert_array_equal(new.q, old.q)
    np.testing.assert_array_equal(new.final_z, old.final_z)
    np.testing.assert_allclose(new.dq_dtheta, old.chi_q, atol=3.0e-16)
    np.testing.assert_allclose(new.Q, old.Q, atol=2.0e-15)
    np.testing.assert_allclose(new.dQ_dtheta, old.chi_Q, atol=2.0e-15)


def test_validation_history_is_immutable_and_shape_checked():
    history = AoTSecondOrderHistory(
        z=np.zeros((2, 3)),
        u=np.ones((2, 3)),
        v=np.full((2, 3), 2.0),
    )
    assert not history.z.flags.writeable
    with pytest.raises(ValueError, match="identical shapes"):
        AoTSecondOrderHistory(
            z=np.zeros((2, 3)),
            u=np.zeros((2, 2)),
            v=np.zeros((2, 3)),
        )


def test_dressel_reference_second_derivative_matches_log_central_difference():
    s = 1.3
    h = 2.0e-4
    reference = dressel_no_drive_reference(s)
    plus = dressel_no_drive_reference(s * np.exp(h)).mean_Q
    minus = dressel_no_drive_reference(s * np.exp(-h)).mean_Q
    numerical = (plus - 2.0 * reference.mean_Q + minus) / (h * h)

    np.testing.assert_allclose(reference.d2Q_dlog_s2, numerical, rtol=2.0e-7)


def test_dressel_second_derivative_is_stable_under_quadrature_refinement():
    coarse = dressel_no_drive_reference(1.3, quadrature_order=96)
    fine = dressel_no_drive_reference(1.3, quadrature_order=128)

    np.testing.assert_allclose(coarse.mean_Q, fine.mean_Q, rtol=0.0, atol=6.0e-13)
    np.testing.assert_allclose(coarse.chi_Q, fine.chi_Q, rtol=0.0, atol=3.0e-11)
    np.testing.assert_allclose(
        coarse.d2Q_dlog_s2,
        fine.d2Q_dlog_s2,
        rtol=0.0,
        atol=4.0e-10,
    )
