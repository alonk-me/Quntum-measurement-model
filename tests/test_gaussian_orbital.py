import numpy as np
import pytest
from scipy.linalg import expm

from quantum_measurement.jw_expansion.gaussian_orbital import (
    apply_orbital_hamiltonian_map,
    article_z_from_covariance,
    chain_stationarity_diagnostics,
    coarsen_standard_normal_noise,
    code_z_from_covariance,
    covariance_tangents_from_orbitals,
    deterministic_phase_qr,
    deterministic_phase_qr_second_order,
    fermion_annihilation_operators,
    gaussian_tangent_growth_metrics,
    ising_bdg_hamiltonian,
    ising_spin_hamiltonian,
    nambu_covariance_from_state,
    particle_hole_swap,
    measurement_qr_tangent_step,
    simulate_exact_small_chain,
    simulate_gaussian_orbital_chain,
    simulate_gaussian_orbital_chain_second_order,
)


def _occupied_state(L):
    state = np.zeros(2**L, dtype=complex)
    state[-1] = 1.0
    return state


@pytest.mark.parametrize("L", [1, 2, 3])
def test_jordan_wigner_number_and_article_spin_convention(L):
    annihilators = fermion_annihilation_operators(L)
    identity = np.eye(2**L)
    for site, annihilator in enumerate(annihilators):
        number = annihilator.conj().T @ annihilator
        factors = [np.eye(2)] * L
        factors[site] = np.diag([1.0, -1.0])
        sigma_z = factors[0]
        for factor in factors[1:]:
            sigma_z = np.kron(sigma_z, factor)
        np.testing.assert_allclose(identity - 2.0 * number, sigma_z)


def test_article_and_legacy_z_are_global_sign_reversals():
    state = np.array([0.6, 0.0, 0.0, 0.8j], dtype=complex)
    covariance = nambu_covariance_from_state(state, L=2)
    article = article_z_from_covariance(covariance, L=2)
    legacy = code_z_from_covariance(covariance, L=2)

    np.testing.assert_allclose(article, -legacy)
    np.testing.assert_allclose(article, np.array([-0.28, -0.28]))


def test_tau_x_is_the_particle_hole_swap_for_declared_nambu_ordering():
    state = np.array([0.6, 0.0, 0.0, 0.8j], dtype=complex)
    covariance = nambu_covariance_from_state(state, L=2)
    tau_x = particle_hole_swap(2)
    identity = np.eye(4)
    zero = np.zeros((2, 2), dtype=complex)
    tau_y = np.block(
        [[zero, -1.0j * np.eye(2)], [1.0j * np.eye(2), zero]]
    )

    assert np.linalg.norm(tau_x @ covariance.conj() @ tau_x + covariance - identity) < 1.0e-14
    assert np.linalg.norm(tau_y @ covariance.conj() @ tau_y + covariance - identity) > 1.0


def test_hand_derived_L1_and_L2_spin_hamiltonians():
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    J = 0.7
    np.testing.assert_allclose(
        ising_spin_hamiltonian(1, J, "open"), np.zeros((2, 2))
    )
    np.testing.assert_allclose(
        ising_spin_hamiltonian(1, J, "periodic"), J * np.eye(2)
    )
    np.testing.assert_allclose(
        ising_spin_hamiltonian(2, J, "open"), J * np.kron(sigma_x, sigma_x)
    )
    np.testing.assert_allclose(
        ising_spin_hamiltonian(2, J, "periodic"),
        2.0 * J * np.kron(sigma_x, sigma_x),
    )


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_bdg_hamiltonian_is_hermitian_and_has_tau_x_symmetry(L, boundary):
    parity = (-1) ** L
    hamiltonian = ising_bdg_hamiltonian(
        L, 0.7, boundary, parity=parity
    )
    tau_x = particle_hole_swap(L)

    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-15)
    np.testing.assert_allclose(
        tau_x @ hamiltonian.conj() @ tau_x,
        -hamiltonian,
        atol=1.0e-15,
    )


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("L", [1, 2, 3])
def test_bdg_covariance_evolution_matches_exact_spin_hamiltonian(L, boundary):
    J = 0.7
    dt = 0.13
    state = _occupied_state(L)
    state_after = expm(-1.0j * ising_spin_hamiltonian(L, J, boundary) * dt) @ state
    exact_covariance = nambu_covariance_from_state(state_after, L)
    covariance = nambu_covariance_from_state(state, L)
    bdg = ising_bdg_hamiltonian(L, J, boundary, parity=(-1) ** L)
    orbital_map = expm(1.0j * bdg * dt)
    mapped_covariance = orbital_map @ covariance @ orbital_map.conj().T

    np.testing.assert_allclose(mapped_covariance, exact_covariance, atol=5.0e-15)


def test_deterministic_qr_has_positive_diagonal_and_rejects_singular_gauge():
    matrix = np.array(
        [[1.0j, 0.2], [1.0, -0.3j], [0.4, 2.0], [-0.1j, 0.5]],
        dtype=complex,
    )
    q_matrix, r_matrix = deterministic_phase_qr(matrix)

    np.testing.assert_allclose(q_matrix @ r_matrix, matrix, atol=1.0e-15)
    np.testing.assert_allclose(q_matrix.conj().T @ q_matrix, np.eye(2), atol=1.0e-15)
    np.testing.assert_allclose(np.imag(np.diag(r_matrix)), 0.0, atol=1.0e-15)
    assert np.all(np.real(np.diag(r_matrix)) > 0.0)

    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        deterministic_phase_qr(np.zeros((4, 2), dtype=complex))


def test_batched_deterministic_qr_matches_individual_cpu_maps():
    matrices = np.random.default_rng(71).standard_normal((3, 6, 3))
    matrices = matrices + 1.0j * np.random.default_rng(72).standard_normal(
        (3, 6, 3)
    )
    batch_q, batch_r = deterministic_phase_qr(matrices)
    individual = [deterministic_phase_qr(matrix) for matrix in matrices]

    np.testing.assert_allclose(batch_q, np.stack([item[0] for item in individual]))
    np.testing.assert_allclose(batch_r, np.stack([item[1] for item in individual]))
    np.testing.assert_allclose(batch_q @ batch_r, matrices, atol=2.0e-15)
    assert np.all(np.real(np.diagonal(batch_r, axis1=-2, axis2=-1)) > 0.0)


def test_deterministic_qr_first_and_second_tangents_match_finite_difference():
    generator = np.random.default_rng(103)
    raw = generator.standard_normal((2, 6, 3)) + 1.0j * generator.standard_normal(
        (2, 6, 3)
    )
    first_raw = 0.2 * (
        generator.standard_normal(raw.shape)
        + 1.0j * generator.standard_normal(raw.shape)
    )
    second_raw = 0.1 * (
        generator.standard_normal(raw.shape)
        + 1.0j * generator.standard_normal(raw.shape)
    )
    q, q1, q2, r, r1, r2 = deterministic_phase_qr_second_order(
        raw, first_raw, second_raw
    )

    np.testing.assert_allclose(first_raw, q1 @ r + q @ r1, atol=2.0e-14)
    np.testing.assert_allclose(
        second_raw, q2 @ r + 2.0 * q1 @ r1 + q @ r2, atol=4.0e-14
    )
    np.testing.assert_allclose(
        np.swapaxes(q.conj(), -2, -1) @ q1
        + np.swapaxes(q1.conj(), -2, -1) @ q,
        0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.swapaxes(q.conj(), -2, -1) @ q2
        + np.swapaxes(q2.conj(), -2, -1) @ q
        + 2.0 * np.swapaxes(q1.conj(), -2, -1) @ q1,
        0.0,
        atol=5.0e-14,
    )

    h = 2.0e-4
    plus_q, plus_r = deterministic_phase_qr(
        raw + h * first_raw + 0.5 * h * h * second_raw
    )
    minus_q, minus_r = deterministic_phase_qr(
        raw - h * first_raw + 0.5 * h * h * second_raw
    )
    np.testing.assert_allclose((plus_q - minus_q) / (2.0 * h), q1, atol=2.0e-8)
    np.testing.assert_allclose(
        (plus_q - 2.0 * q + minus_q) / (h * h), q2, atol=2.0e-7
    )
    np.testing.assert_allclose((plus_r - minus_r) / (2.0 * h), r1, atol=2.0e-8)
    np.testing.assert_allclose(
        (plus_r - 2.0 * r + minus_r) / (h * h), r2, atol=2.0e-7
    )


def test_qr_tangent_path_rejects_ill_conditioned_gauge():
    zero = np.zeros((4, 2), dtype=complex)
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        deterministic_phase_qr_second_order(zero, zero, zero)


@pytest.mark.parametrize("L", [1, 2])
def test_hamiltonian_map_propagates_both_orbital_tangent_orders(L):
    generator = np.random.default_rng(210 + L)
    orbitals = generator.standard_normal((2 * L, L)).astype(complex)
    first = generator.standard_normal((2 * L, L)).astype(complex)
    second = generator.standard_normal((2 * L, L)).astype(complex)
    bdg = ising_bdg_hamiltonian(L, 0.8, "open", parity=(-1) ** L)
    orbital_map = expm(1.0j * bdg * 0.03)

    mapped = apply_orbital_hamiltonian_map(orbital_map, orbitals, first, second)
    np.testing.assert_allclose(mapped[0], orbital_map @ orbitals)
    np.testing.assert_allclose(mapped[1], orbital_map @ first)
    np.testing.assert_allclose(mapped[2], orbital_map @ second)


def test_measurement_qr_tangents_match_common_noise_finite_difference():
    generator = np.random.default_rng(301)
    raw = generator.standard_normal((6, 3)) + 1.0j * generator.standard_normal((6, 3))
    orbitals, _ = deterministic_phase_qr(raw)
    first = 0.1 * (
        generator.standard_normal((6, 3))
        + 1.0j * generator.standard_normal((6, 3))
    )
    second = 0.05 * (
        generator.standard_normal((6, 3))
        + 1.0j * generator.standard_normal((6, 3))
    )
    noise = generator.standard_normal(3)
    epsilon = 0.12
    q, q1, q2, _, _, _ = measurement_qr_tangent_step(
        orbitals, first, second, noise, epsilon
    )

    h = 1.0e-4
    zero = np.zeros_like(orbitals)
    plus_orbitals = orbitals + h * first + 0.5 * h * h * second
    minus_orbitals = orbitals - h * first + 0.5 * h * h * second
    plus = measurement_qr_tangent_step(
        plus_orbitals, zero, zero, noise, epsilon * np.exp(0.5 * h)
    )[0]
    minus = measurement_qr_tangent_step(
        minus_orbitals, zero, zero, noise, epsilon * np.exp(-0.5 * h)
    )[0]

    np.testing.assert_allclose((plus - minus) / (2.0 * h), q1, atol=2.0e-8)
    np.testing.assert_allclose(
        (plus - 2.0 * q + minus) / (h * h), q2, atol=3.0e-7
    )

    covariance, first_covariance, second_covariance = (
        covariance_tangents_from_orbitals(q, q1, q2)
    )
    assert covariance.shape == first_covariance.shape == second_covariance.shape == (6, 6)


def test_tangent_growth_metrics_identify_pure_orbital_gauge_motion():
    orbitals = np.eye(4, 2, dtype=complex)
    generator = np.array([[0.0, 0.7], [-0.7, 0.0]], dtype=complex)
    first = orbitals @ generator
    second = orbitals @ generator @ generator

    metrics = gaussian_tangent_growth_metrics(orbitals, first, second)

    assert metrics["first_orbital_norm"] > 0.0
    assert metrics["second_orbital_norm"] > 0.0
    assert metrics["first_within_occupied_norm"] > 0.0
    assert metrics["second_within_occupied_norm"] > 0.0
    np.testing.assert_allclose(metrics["first_horizontal_norm"], 0.0)
    np.testing.assert_allclose(metrics["second_horizontal_norm"], 0.0)
    np.testing.assert_allclose(metrics["first_covariance_norm"], 0.0)
    np.testing.assert_allclose(metrics["second_covariance_norm"], 0.0)
    assert metrics["second_covariance_linear_norm"] > 0.0
    assert metrics["second_covariance_quadratic_norm"] > 0.0


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("L", [1, 2, 4])
def test_exact_and_orbital_complete_histories_agree(L, boundary):
    noise = np.random.default_rng(100 + L).standard_normal((2, 7, L))
    arguments = dict(
        L=L,
        gamma=1.3,
        J=0.8,
        dt=0.01,
        n_burnin=2,
        n_samples=5,
        boundary=boundary,
        noise=noise,
        store_covariance_history=True,
    )
    exact = simulate_exact_small_chain(**arguments)
    orbital = simulate_gaussian_orbital_chain(**arguments)

    np.testing.assert_allclose(orbital.z_history, exact.z_history, atol=4.0e-14)
    np.testing.assert_allclose(
        orbital.covariance_history, exact.covariance_history, atol=4.0e-14
    )
    np.testing.assert_allclose(orbital.q, exact.q, atol=2.0e-14)
    np.testing.assert_allclose(orbital.C0, exact.C0, atol=2.0e-14)
    np.testing.assert_allclose(orbital.q + orbital.C0, 2.0, atol=1.0e-15)
    assert orbital.noise_hash == exact.noise_hash
    assert orbital.parity == exact.parity == (-1) ** L


def test_orbital_path_preserves_pure_gaussian_invariants_without_repairs():
    result = simulate_gaussian_orbital_chain(
        L=4,
        gamma=4.0,
        J=1.0,
        dt=0.005,
        n_burnin=20,
        n_samples=80,
        boundary="periodic",
        n_trajectories=3,
        seed=20260905,
    )

    assert np.max(result.max_hermiticity_residual) < 1.0e-14
    assert np.max(result.max_particle_hole_residual) < 1.0e-13
    assert np.max(result.max_projector_residual) < 1.0e-14
    assert np.max(result.max_trace_residual) < 1.0e-14
    assert np.max(result.max_eigenvalue_residual) < 1.0e-14
    assert np.max(result.max_qr_residual) < 1.0e-14
    assert result.integrator == "pure_gaussian_orbital_exponential_split_v1"


def test_gaussian_noise_contract_seed_replay_and_shapes():
    arguments = dict(
        L=2,
        gamma=1.0,
        J=1.0,
        dt=0.01,
        n_burnin=1,
        n_samples=3,
        boundary="open",
    )
    first = simulate_gaussian_orbital_chain(
        **arguments, n_trajectories=2, seed=17
    )
    second = simulate_gaussian_orbital_chain(
        **arguments, n_trajectories=2, seed=17
    )
    np.testing.assert_array_equal(first.z_history, second.z_history)
    assert first.noise_kind == "standard_normal"
    assert first.noise_shape == (2, 4, 2)
    assert first.noise_hash == second.noise_hash

    one_noise = np.zeros((4, 2))
    one = simulate_gaussian_orbital_chain(**arguments, noise=one_noise)
    assert one.z_history.shape == (1, 5, 2)

    with pytest.raises(ValueError, match="exactly one"):
        simulate_gaussian_orbital_chain(**arguments)
    with pytest.raises(ValueError, match="exactly one"):
        simulate_gaussian_orbital_chain(**arguments, noise=one_noise, seed=1)
    with pytest.raises(ValueError, match="noise must have shape"):
        simulate_gaussian_orbital_chain(**arguments, noise=np.zeros((3, 2)))
    malformed = one_noise.copy()
    malformed[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulate_gaussian_orbital_chain(**arguments, noise=malformed)


def test_batch_size_one_and_explicit_single_noise_are_identical():
    noise = np.random.default_rng(44).standard_normal((6, 2))
    arguments = dict(
        L=2,
        gamma=0.7,
        J=1.0,
        dt=0.01,
        n_burnin=1,
        n_samples=5,
        boundary="periodic",
    )
    single = simulate_gaussian_orbital_chain(**arguments, noise=noise)
    batched = simulate_gaussian_orbital_chain(
        **arguments, noise=noise[np.newaxis, :, :]
    )

    np.testing.assert_array_equal(single.z_history, batched.z_history)
    np.testing.assert_array_equal(single.q, batched.q)
    np.testing.assert_array_equal(single.final_covariance, batched.final_covariance)


def test_base_orbital_observable_converges_under_coupled_timestep_refinement():
    L = 2
    fine_dt = 0.000625
    fine_steps = 320
    fine_noise = np.random.default_rng(123).standard_normal((4, fine_steps, L))
    results = []
    for factor in (16, 8, 4, 2, 1):
        noise = coarsen_standard_normal_noise(fine_noise, factor)
        results.append(
            simulate_gaussian_orbital_chain(
                L=L,
                gamma=2.0,
                J=1.0,
                dt=fine_dt * factor,
                n_burnin=0,
                n_samples=fine_steps // factor,
                boundary="open",
                noise=noise,
            )
        )

    finest = results[-1]
    q_errors = np.array(
        [
            np.sqrt(np.mean((result.q - finest.q) ** 2))
            for result in results[:-1]
        ]
    )
    np.testing.assert_array_less(q_errors[1:], q_errors[:-1])
    np.testing.assert_array_less(1.5, q_errors[:-1] / q_errors[1:])


def test_stationarity_diagnostics_preserve_trajectory_and_site_axes():
    result = simulate_gaussian_orbital_chain(
        L=2,
        gamma=1.3,
        J=1.0,
        dt=0.01,
        n_burnin=2,
        n_samples=6,
        n_trajectories=3,
        boundary="open",
        seed=87,
    )
    diagnostics = chain_stationarity_diagnostics(
        result,
        start_step=result.n_burnin + 1,
        n_windows=3,
        max_lag=2,
    )

    assert diagnostics.window_mean_z2.shape == (3, 3)
    assert diagnostics.window_site_mean_z2.shape == (3, 3, 2)
    np.testing.assert_array_equal(diagnostics.window_start_steps, [3, 5, 7])
    np.testing.assert_array_equal(diagnostics.window_stop_steps, [5, 7, 9])
    np.testing.assert_allclose(
        1.0 + np.mean(diagnostics.window_mean_z2, axis=1), result.q
    )
    np.testing.assert_allclose(
        np.mean(diagnostics.window_site_mean_z2, axis=2),
        diagnostics.window_mean_z2,
    )
    assert np.all(diagnostics.autocorrelation_time_steps >= 1.0)
    assert np.all(diagnostics.effective_sample_size <= result.n_samples)
    assert not diagnostics.window_mean_z2.flags.writeable


def test_stationarity_diagnostics_handle_constant_series_and_validate_segment():
    result = simulate_gaussian_orbital_chain(
        L=1,
        gamma=0.5,
        J=1.0,
        dt=0.02,
        n_burnin=1,
        n_samples=5,
        boundary="periodic",
        noise=np.zeros((6, 1)),
    )
    diagnostics = chain_stationarity_diagnostics(
        result, start_step=2, n_windows=2
    )

    np.testing.assert_array_equal(diagnostics.window_mean_z2, np.ones((1, 2)))
    np.testing.assert_array_equal(diagnostics.autocorrelation_time_steps, [1.0])
    np.testing.assert_array_equal(diagnostics.effective_sample_size, [5.0])
    np.testing.assert_array_equal(diagnostics.autocorrelation_time, [0.02])

    with pytest.raises(ValueError, match="start_step"):
        chain_stationarity_diagnostics(result, start_step=7)
    with pytest.raises(ValueError, match="n_windows"):
        chain_stationarity_diagnostics(result, start_step=2, n_windows=6)
    with pytest.raises(ValueError, match="max_lag"):
        chain_stationarity_diagnostics(result, start_step=2, max_lag=5)


@pytest.mark.parametrize("L", [1, 2])
def test_second_order_chain_matches_common_noise_finite_difference(L):
    gamma = 1.3
    J = 0.8
    dt = 0.01
    n_burnin = 3
    n_samples = 9
    noise = np.random.default_rng(500 + L).standard_normal((2, 12, L))
    arguments = dict(
        L=L,
        gamma=gamma,
        J=J,
        dt=dt,
        n_burnin=n_burnin,
        n_samples=n_samples,
        boundary="periodic",
        noise=noise,
    )
    tangent, history = simulate_gaussian_orbital_chain_second_order(
        **arguments, store_history=True
    )
    central = simulate_exact_small_chain(**arguments)
    h = 0.002
    plus_arguments = dict(arguments, gamma=gamma * np.exp(h))
    minus_arguments = dict(arguments, gamma=gamma * np.exp(-h))
    plus = simulate_exact_small_chain(**plus_arguments)
    minus = simulate_exact_small_chain(**minus_arguments)

    finite_difference_u = (plus.z_history - minus.z_history) / (2.0 * h)
    finite_difference_v = (
        plus.z_history - 2.0 * central.z_history + minus.z_history
    ) / (h * h)
    retained = slice(n_burnin + 1, None)
    np.testing.assert_allclose(
        tangent.q,
        np.mean(1.0 + np.square(history.z[:, retained, :]), axis=(1, 2)),
    )
    np.testing.assert_allclose(
        tangent.dq_dtheta,
        np.mean(
            2.0 * history.z[:, retained, :] * history.u[:, retained, :],
            axis=(1, 2),
        ),
    )
    np.testing.assert_allclose(
        tangent.d2q_dtheta2,
        np.mean(
            2.0
            * (
                np.square(history.u[:, retained, :])
                + history.z[:, retained, :] * history.v[:, retained, :]
            ),
            axis=(1, 2),
        ),
    )
    assert np.all((1.0 <= tangent.q) & (tangent.q <= 2.0))
    np.testing.assert_allclose(history.z, central.z_history, atol=3.0e-15)
    np.testing.assert_allclose(history.u, finite_difference_u, atol=2.0e-8)
    np.testing.assert_allclose(history.v, finite_difference_v, atol=2.0e-8)
    np.testing.assert_allclose(
        tangent.dq_dtheta, (plus.q - minus.q) / (2.0 * h), atol=1.0e-8
    )
    np.testing.assert_allclose(
        tangent.d2q_dtheta2,
        (plus.q - 2.0 * central.q + minus.q) / (h * h),
        atol=1.0e-8,
    )
    averaging_time = n_samples * dt
    plus_Q = plus.gamma * L * averaging_time * plus.q
    central_Q = gamma * L * averaging_time * central.q
    minus_Q = minus.gamma * L * averaging_time * minus.q
    np.testing.assert_allclose(
        tangent.dQ_dtheta, (plus_Q - minus_Q) / (2.0 * h), atol=5.0e-7
    )
    np.testing.assert_allclose(
        tangent.d2Q_dtheta2,
        (plus_Q - 2.0 * central_Q + minus_Q) / (h * h),
        atol=5.0e-7,
    )
    np.testing.assert_allclose(
        tangent.final_first_covariance,
        (plus.final_covariance - minus.final_covariance) / (2.0 * h),
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        tangent.final_second_covariance,
        (
            plus.final_covariance
            - 2.0 * central.final_covariance
            + minus.final_covariance
        )
        / (h * h),
        atol=2.0e-8,
    )
    assert history.growth is not None
    assert history.growth.first_orbital_norm.shape == (2, 13)
    assert history.growth.invariant_residuals.shape == (2, 13, 4, 3)
    assert np.all(np.isfinite(history.growth.minimum_qr_diagonal))
    with pytest.raises(ValueError, match="read-only"):
        history.growth.second_covariance_norm[0, 0] = 0.0


def test_second_order_chain_result_schema_invariants_and_no_repair_policy():
    result, history = simulate_gaussian_orbital_chain_second_order(
        L=4,
        gamma=4.0,
        J=1.0,
        dt=0.005,
        n_burnin=5,
        n_samples=15,
        boundary="periodic",
        n_trajectories=3,
        seed=20260906,
        store_history=False,
    )

    assert history is None
    assert result.q.shape == (3,)
    assert result.final_orbitals.shape == (3, 8, 4)
    assert result.final_covariance.shape == (3, 8, 8)
    assert result.max_hermiticity_residual.shape == (3, 3)
    assert np.max(result.max_hermiticity_residual) < 1.0e-14
    assert np.max(result.max_particle_hole_residual) < 2.0e-13
    assert np.max(result.max_projector_residual) < 2.0e-13
    assert np.max(result.max_orbital_constraint_residual) < 2.0e-13
    assert np.min(result.minimum_qr_diagonal) > 1.0e-14
    np.testing.assert_array_equal(result.repair_count, np.zeros(3))
    assert result.repair_policy == "none"
    assert not result.repair_fired
    assert not result.fallback_fired
    assert not result.clipping_fired
    assert result.derivative_parameter == "log_gamma_at_fixed_J"
    assert result.log_g == pytest.approx(0.0)
    with pytest.raises(ValueError, match="read-only"):
        result.q[0] = 0.0


def test_second_order_single_site_article_sign_and_extensive_prefactor():
    gamma = 2.5
    dt = 0.01
    n_samples = 8
    result, history = simulate_gaussian_orbital_chain_second_order(
        L=1,
        gamma=gamma,
        J=1.0,
        dt=dt,
        n_burnin=2,
        n_samples=n_samples,
        boundary="periodic",
        noise=np.random.default_rng(900).standard_normal((10, 1)),
        store_history=True,
    )

    np.testing.assert_allclose(history.z, -1.0, atol=2.0e-15)
    np.testing.assert_allclose(history.u, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(history.v, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(result.q, [2.0], atol=2.0e-15)
    np.testing.assert_allclose(result.dq_dtheta, [0.0], atol=2.0e-15)
    np.testing.assert_allclose(result.d2q_dtheta2, [0.0], atol=2.0e-15)
    expected_Q = 2.0 * gamma * n_samples * dt
    np.testing.assert_allclose(result.Q, [expected_Q], atol=2.0e-15)
    np.testing.assert_allclose(result.dQ_dtheta, [expected_Q], atol=2.0e-15)
    np.testing.assert_allclose(result.d2Q_dtheta2, [expected_Q], atol=2.0e-15)


def test_exact_reference_derivative_errors_show_second_order_h_convergence():
    L = 2
    gamma = 4.0
    noise = np.random.default_rng(908).standard_normal((4, 40, L))
    arguments = dict(
        L=L,
        gamma=gamma,
        J=1.0,
        dt=0.005,
        n_burnin=5,
        n_samples=35,
        boundary="periodic",
        noise=noise,
    )
    tangent, history = simulate_gaussian_orbital_chain_second_order(
        **arguments, store_history=True
    )
    central = simulate_exact_small_chain(**arguments)
    errors = []
    for h in (0.04, 0.02, 0.01):
        plus = simulate_exact_small_chain(
            **dict(arguments, gamma=gamma * np.exp(h))
        )
        minus = simulate_exact_small_chain(
            **dict(arguments, gamma=gamma * np.exp(-h))
        )
        finite_difference_u = (plus.z_history - minus.z_history) / (2.0 * h)
        finite_difference_v = (
            plus.z_history - 2.0 * central.z_history + minus.z_history
        ) / (h * h)
        errors.append(
            (
                np.sqrt(np.mean(np.square(finite_difference_u - history.u))),
                np.sqrt(np.mean(np.square(finite_difference_v - history.v))),
                np.sqrt(
                    np.mean(
                        np.square(
                            (plus.q - minus.q) / (2.0 * h)
                            - tangent.dq_dtheta
                        )
                    )
                ),
                np.sqrt(
                    np.mean(
                        np.square(
                            (plus.q - 2.0 * central.q + minus.q) / (h * h)
                            - tangent.d2q_dtheta2
                        )
                    )
                ),
            )
        )
    errors = np.asarray(errors)
    np.testing.assert_array_less(errors[1:], errors[:-1])
    np.testing.assert_array_less(3.5, errors[:-1] / errors[1:])


def test_second_order_observables_refine_on_coupled_brownian_path():
    L = 2
    fine_dt = 0.00125
    fine_steps = 160
    fine_noise = np.random.default_rng(991).standard_normal((4, fine_steps, L))
    results = []
    for factor in (4, 2, 1):
        result, _ = simulate_gaussian_orbital_chain_second_order(
            L=L,
            gamma=0.4,
            J=1.0,
            dt=fine_dt * factor,
            n_burnin=0,
            n_samples=fine_steps // factor,
            boundary="periodic",
            noise=coarsen_standard_normal_noise(fine_noise, factor),
        )
        results.append(result)

    finest = results[-1]
    for field in ("q", "dq_dtheta", "d2q_dtheta2"):
        errors = [
            np.sqrt(np.mean(np.square(getattr(result, field) - getattr(finest, field))))
            for result in results[:-1]
        ]
        assert errors[1] < errors[0]


def test_second_order_chain_rejects_ambiguous_noise_and_replays_seed():
    arguments = dict(
        L=2,
        gamma=1.0,
        J=1.0,
        dt=0.01,
        n_burnin=1,
        n_samples=3,
        boundary="open",
    )
    first, _ = simulate_gaussian_orbital_chain_second_order(
        **arguments, n_trajectories=2, seed=19
    )
    second, _ = simulate_gaussian_orbital_chain_second_order(
        **arguments, n_trajectories=2, seed=19
    )
    np.testing.assert_array_equal(first.q, second.q)
    np.testing.assert_array_equal(
        first.final_second_orbital_tangent,
        second.final_second_orbital_tangent,
    )
    assert first.noise_hash == second.noise_hash

    with pytest.raises(ValueError, match="exactly one"):
        simulate_gaussian_orbital_chain_second_order(**arguments)
    with pytest.raises(ValueError, match="exactly one"):
        simulate_gaussian_orbital_chain_second_order(
            **arguments, noise=np.zeros((4, 2)), seed=19
        )


def test_noise_coarsening_preserves_brownian_increment():
    fine = np.arange(24, dtype=float).reshape(2, 6, 2)
    coarse = coarsen_standard_normal_noise(fine, 3)
    np.testing.assert_allclose(
        np.sqrt(3.0) * coarse,
        fine.reshape(2, 2, 3, 2).sum(axis=2),
    )
    with pytest.raises(ValueError, match="divide"):
        coarsen_standard_normal_noise(fine, 4)
