import numpy as np
import pytest
from scipy.linalg import expm

from quantum_measurement.jw_expansion.gaussian_orbital import (
    article_z_from_covariance,
    chain_stationarity_diagnostics,
    coarsen_standard_normal_noise,
    code_z_from_covariance,
    deterministic_phase_qr,
    fermion_annihilation_operators,
    ising_bdg_hamiltonian,
    ising_spin_hamiltonian,
    nambu_covariance_from_state,
    particle_hole_swap,
    simulate_exact_small_chain,
    simulate_gaussian_orbital_chain,
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


def test_noise_coarsening_preserves_brownian_increment():
    fine = np.arange(24, dtype=float).reshape(2, 6, 2)
    coarse = coarsen_standard_normal_noise(fine, 3)
    np.testing.assert_allclose(
        np.sqrt(3.0) * coarse,
        fine.reshape(2, 2, 3, 2).sum(axis=2),
    )
    with pytest.raises(ValueError, match="divide"):
        coarsen_standard_normal_noise(fine, 4)
