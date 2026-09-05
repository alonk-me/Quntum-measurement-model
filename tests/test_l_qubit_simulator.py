"""Tests for LQubitCorrelationSimulator in jw_expansion/l_qubit_correlation_simulator.py."""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_expansion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from l_qubit_correlation_simulator import LQubitCorrelationSimulator
from run_z2_scan import get_time_params


def validate_stable_evolution(gamma: float, L: int = 16) -> Dict[str, float]:
    """Run one stable trajectory and validate projector/BdG constraints."""
    _, dt, _, _ = get_time_params(gamma)
    n_steps = 512
    T = float(dt * n_steps)
    epsilon = float(np.sqrt(gamma * dt))

    sim = LQubitCorrelationSimulator(
        L=L,
        J=1.0,
        epsilon=epsilon,
        N_steps=n_steps,
        T=T,
        closed_boundary=True,
        device="cpu",
        use_stable_integrator=True,
        stable_projector_enforce=True,
        bdg_enforce_threshold=0.0,
        enable_stability_monitor=True,
        rng=np.random.default_rng(1234),
    )

    _, z_traj, _ = sim.simulate_trajectory()
    monitor = sim.get_stability_monitor_data()
    z2 = float(np.mean(z_traj[-1] ** 2))

    return {
        "projector_max": float(np.max(monitor["projector"])) if monitor["projector"].size else float("inf"),
        "bdg_max": float(np.max(monitor["bdg"])) if monitor["bdg"].size else float("inf"),
        "z2": z2,
        "z2_is_finite": float(np.isfinite(z2)),
    }


class TestLQubitInitialization:
    """Tests for LQubitCorrelationSimulator initialization."""

    def test_default_params(self):
        sim = LQubitCorrelationSimulator()
        assert sim.L == 2
        assert sim.J == 1.0
        assert sim.epsilon == 0.1
        assert sim.N_steps == 1000

    def test_invalid_L_raises(self):
        with pytest.raises(ValueError, match="L must be at least 1"):
            LQubitCorrelationSimulator(L=0)

    def test_invalid_N_steps_raises(self):
        with pytest.raises(ValueError, match="N_steps must be at least 1"):
            LQubitCorrelationSimulator(N_steps=0)

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            LQubitCorrelationSimulator(T=-1.0)

    def test_dt_computed(self):
        sim = LQubitCorrelationSimulator(T=2.0, N_steps=200)
        assert sim.dt == pytest.approx(0.01)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            LQubitCorrelationSimulator(epsilon=0.0)

    def test_rng_created(self):
        sim = LQubitCorrelationSimulator(rng=None)
        assert sim.rng is not None

    def test_accepts_device_cpu(self):
        sim = LQubitCorrelationSimulator(device="cpu", N_steps=5, rng=np.random.default_rng(1))
        Q, z_traj, xi_traj = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z_traj.shape == (6, sim.L)
        assert xi_traj.shape == (5, sim.L)


class TestBuildHamiltonian:
    """Tests for _build_hamiltonian."""

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_shape(self, L):
        sim = LQubitCorrelationSimulator(L=L)
        assert sim.h.shape == (2 * L, 2 * L)

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_hermitian(self, L):
        sim = LQubitCorrelationSimulator(L=L, J=1.0)
        h = sim.h
        # BdG Hamiltonian is not necessarily Hermitian, but let's check it is
        # at least complex-valued with correct shape
        assert h.shape == (2 * L, 2 * L)

    def test_open_vs_periodic(self):
        sim_open = LQubitCorrelationSimulator(L=3, J=1.0, closed_boundary=False)
        sim_pbc = LQubitCorrelationSimulator(L=3, J=1.0, closed_boundary=True)
        # Periodic boundary adds corner element, so matrices should differ
        assert not np.allclose(sim_open.h, sim_pbc.h)

    def test_j_zero_gives_zero_hamiltonian(self):
        sim = LQubitCorrelationSimulator(L=2, J=0.0)
        assert np.allclose(sim.h, 0.0)


class TestComputeZValues:
    """Tests for _compute_z_values."""

    def test_shape(self):
        sim = LQubitCorrelationSimulator(L=3)
        G = sim.G_initial.copy()
        z = sim._compute_z_values(G)
        assert z.shape == (3,)

    def test_initial_state_all_up(self):
        """Initial state G_initial has G[i,i]=1 for i<L, so z[i] = 2*1 - 1 = 1."""
        sim = LQubitCorrelationSimulator(L=2)
        z = sim._compute_z_values(sim.G_initial)
        assert np.allclose(z, 1.0)

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_formula_z_equals_2Re_G_diag_minus_1(self, L):
        """Verify z[i] = 2*Re(G[i,i]) - 1."""
        sim = LQubitCorrelationSimulator(L=L)
        G = sim.G_initial.copy()
        # Perturb G slightly
        G[0, 0] = 0.7 + 0.0j
        z = sim._compute_z_values(G)
        assert z[0] == pytest.approx(2.0 * 0.7 - 1.0)


class TestHamiltonianStep:
    """Tests for _hamiltonian_step."""

    def test_preserves_hermiticity(self, minimal_params_correlation):
        """_hamiltonian_step adds -2i*dt*(GH-HG) to G; the anti-Hermitian part is O(dt)."""
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        G_new = sim._hamiltonian_step(G)
        anti_herm = G_new - G_new.conj().T
        assert np.all(np.abs(anti_herm) < 0.5)  # small, proportional to dt

    def test_j_zero_no_change(self):
        sim = LQubitCorrelationSimulator(L=2, J=0.0, N_steps=100, T=1.0)
        G = sim.G_initial.copy()
        G_new = sim._hamiltonian_step(G)
        assert np.allclose(G, G_new)


class TestMeasurementStep:
    """Tests for _measurement_step."""

    def test_diagonal_clipped(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        G_new, _ = sim._measurement_step(G)
        diag = np.real(np.diag(G_new))
        assert np.all(diag >= -1e-12)
        assert np.all(diag <= 1.0 + 1e-12)

    def test_hermitian_after_step(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        G_new, _ = sim._measurement_step(G)
        assert np.allclose(G_new, G_new.conj().T, atol=1e-10)

    def test_outcomes_binary(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        _, xi = sim._measurement_step(G)
        assert set(xi).issubset({-1, 1})

    def test_outcomes_shape(self):
        sim = LQubitCorrelationSimulator(L=4)
        G = sim.G_initial.copy()
        _, xi = sim._measurement_step(G)
        assert xi.shape == (4,)


class TestSimulateTrajectory:
    """Tests for simulate_trajectory."""

    @pytest.mark.parametrize("L,closed", [(2, False), (3, False), (5, True)])
    def test_return_shapes(self, L, closed):
        sim = LQubitCorrelationSimulator(
            L=L, J=1.0, epsilon=0.1, N_steps=50, T=1.0,
            closed_boundary=closed, rng=np.random.default_rng(42),
        )
        Q, z_traj, xi_traj = sim.simulate_trajectory()
        assert z_traj.shape == (51, L)
        assert xi_traj.shape == (50, L)

    def test_q_is_finite(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q, _, _ = sim.simulate_trajectory()
        assert np.isfinite(Q)

    def test_z_trajectory_in_range(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        _, z_traj, _ = sim.simulate_trajectory()
        assert np.all(z_traj >= -1.0 - 1e-10)
        assert np.all(z_traj <= 1.0 + 1e-10)

    def test_entropy_sign(self, minimal_params_correlation):
        """Entropy production Q can be positive or negative but should be finite."""
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q, _, _ = sim.simulate_trajectory()
        assert np.isfinite(Q)

    def test_batch_size_one_parity(self, minimal_params_correlation):
        """Batch n=1 should match serial trajectory for the same xi sequence."""
        params = dict(minimal_params_correlation)
        seed = 123
        params["rng"] = np.random.default_rng(seed)
        sim = LQubitCorrelationSimulator(**params)

        Q_serial, z_serial, xi_serial = sim.simulate_trajectory()

        rng_for_xi = np.random.default_rng(seed)
        xi_batch = rng_for_xi.choice([-1, 1], size=(1, sim.N_steps, sim.L)).astype(np.int8)
        Q_batch, z_batch, xi_out = sim.simulate_trajectory_batch(1, xi_batch=xi_batch)

        assert np.allclose(Q_batch[0], Q_serial, atol=1e-12)
        assert np.allclose(z_batch[0], z_serial, atol=1e-12)
        assert np.array_equal(xi_out[0], xi_serial)

    def test_batch_shapes(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q, z, xi = sim.simulate_trajectory_batch(3)
        assert Q.shape == (3,)
        assert z.shape == (3, sim.N_steps + 1, sim.L)
        assert xi.shape == (3, sim.N_steps, sim.L)


class TestSimulateEnsemble:
    """Tests for simulate_ensemble."""

    def test_shapes(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        n_traj = 5
        Q_vals, z_series, xi_series = sim.simulate_ensemble(n_traj)
        assert Q_vals.shape == (n_traj,)
        assert z_series.shape == (n_traj, sim.N_steps + 1, sim.L)
        assert xi_series.shape == (n_traj, sim.N_steps, sim.L)

    def test_q_all_finite(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q_vals, _, _ = sim.simulate_ensemble(5)
        assert np.all(np.isfinite(Q_vals))

    def test_ensemble_batch_shapes(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q_vals, z_series, xi_series = sim.simulate_ensemble(7, batch_size=3)
        assert Q_vals.shape == (7,)
        assert z_series.shape == (7, sim.N_steps + 1, sim.L)
        assert xi_series.shape == (7, sim.N_steps, sim.L)


class TestSimulateZ2Ensemble:
    """Tests for simulate_z2_mean_ensemble."""

    def test_single_trajectory_matches_serial_z2(self):
        sim = LQubitCorrelationSimulator(
            L=3,
            J=1.0,
            epsilon=0.1,
            N_steps=40,
            T=1.0,
            closed_boundary=True,
            rng=np.random.default_rng(123),
        )
        if hasattr(sim.backend, "seed"):
            sim.backend.seed(123)

        z2_single = sim.simulate_z2_mean()

        sim2 = LQubitCorrelationSimulator(
            L=3,
            J=1.0,
            epsilon=0.1,
            N_steps=40,
            T=1.0,
            closed_boundary=True,
            rng=np.random.default_rng(123),
        )
        if hasattr(sim2.backend, "seed"):
            sim2.backend.seed(123)
        z2_ensemble = sim2.simulate_z2_mean_ensemble(n_trajectories=1, batch_size=1)

        assert np.isfinite(z2_single)
        assert np.isfinite(z2_ensemble)
        assert z2_ensemble == pytest.approx(z2_single, rel=1e-8, abs=1e-8)

    def test_ensemble_uncertainty_outputs(self):
        sim = LQubitCorrelationSimulator(
            L=3,
            J=1.0,
            epsilon=0.1,
            N_steps=30,
            T=1.0,
            closed_boundary=True,
            rng=np.random.default_rng(7),
        )
        if hasattr(sim.backend, "seed"):
            sim.backend.seed(7)

        mean, std, stderr = sim.simulate_z2_mean_ensemble(
            n_trajectories=5,
            batch_size=2,
            return_std_err=True,
        )

        assert np.isfinite(mean)
        assert np.isfinite(std)
        assert np.isfinite(stderr)
        assert std >= 0.0
        assert stderr >= 0.0

    def test_simulate_z2_mean_returns_nan_on_non_finite_z(self):
        sim = LQubitCorrelationSimulator(
            L=3,
            J=1.0,
            epsilon=0.1,
            N_steps=5,
            T=1.0,
            closed_boundary=True,
            rng=np.random.default_rng(11),
        )

        original_compute = sim._compute_z_values
        calls = {"count": 0}

        def _patched_compute(G):
            calls["count"] += 1
            if calls["count"] >= 2:
                return np.array([np.nan] * sim.L, dtype=float)
            return original_compute(G)

        sim._compute_z_values = _patched_compute  # type: ignore[method-assign]
        out = sim.simulate_z2_mean()
        assert np.isnan(out)


class TestLegacyDirectCorrelationIntegrator:
    """Regression tests for the superseded direct-G path.

    Interior eigenvalues and repair behavior checked here are legacy behavior,
    not acceptance criteria for the pure orbital/QR trajectory implementation.
    """

    @pytest.mark.parametrize("L", [8, 16, 32])
    @pytest.mark.parametrize("gamma", [0.5, 2.0, 5.0])
    def test_z2_not_saturated(self, L, gamma):
        _, dt, _, _ = get_time_params(gamma)
        n_steps = 256
        T = float(dt * n_steps)
        epsilon = float(np.sqrt(gamma * dt))

        sim = LQubitCorrelationSimulator(
            L=L,
            J=1.0,
            epsilon=epsilon,
            N_steps=n_steps,
            T=T,
            closed_boundary=True,
            device="cpu",
            use_stable_integrator=True,
            enable_stability_monitor=True,
            rng=np.random.default_rng(2026),
        )

        z2 = sim.simulate_z2_mean()
        assert np.isfinite(z2)
        assert 0.01 < z2 < 1.99

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_validate_stable_evolution_gamma_sweep(self, gamma):
        result = validate_stable_evolution(gamma=gamma, L=16)
        assert result["projector_max"] < 1e-8
        assert result["bdg_max"] < 1e-8
        assert bool(result["z2_is_finite"])
        assert 0.01 < result["z2"] < 1.99

    def test_stable_vs_euler_agree_small_dt(self):
        L = 3
        dt = 1e-5
        n_steps = 10
        T = dt * n_steps
        gamma = 0.5
        epsilon = float(np.sqrt(gamma * dt))

        sim_euler = LQubitCorrelationSimulator(
            L=L,
            J=0.1,
            epsilon=epsilon,
            N_steps=n_steps,
            T=T,
            closed_boundary=True,
            device="cpu",
            use_stable_integrator=False,
            rng=np.random.default_rng(11),
        )
        sim_stable = LQubitCorrelationSimulator(
            L=L,
            J=0.1,
            epsilon=epsilon,
            N_steps=n_steps,
            T=T,
            closed_boundary=True,
            device="cpu",
            use_stable_integrator=True,
            stable_projector_enforce=False,
            bdg_enforce_threshold=1e9,
            rng=np.random.default_rng(11),
        )

        G_euler = np.asarray(sim_euler.G_initial, dtype=complex)[None, :, :]
        G_stable = np.asarray(sim_stable.G_initial, dtype=complex)[None, :, :]
        xi_rng = np.random.default_rng(42)

        for _ in range(n_steps):
            G_euler = sim_euler._hamiltonian_step_batch(G_euler)
            xi_step = xi_rng.choice([-1, 1], size=(1, L)).astype(np.int8)
            G_euler = sim_euler._measurement_step_batch(G_euler, xi_step, apply_projection=False)
            G_euler = sim_euler._enforce_hermiticity_batch(G_euler)

            G_stable = sim_stable._stable_step_batch(G_stable, xi_step)

        diff = np.linalg.norm(G_euler[0] - G_stable[0], ord="fro")
        assert diff < 1e-8

    def test_no_mixed_state_drift(self):
        """Keep the historical mixed-state drift metric as a regression only."""
        gamma = 2.0
        dt = 1e-3
        n_steps = 500
        epsilon = float(np.sqrt(gamma * dt))

        sim = LQubitCorrelationSimulator(
            L=16,
            J=1.0,
            epsilon=epsilon,
            N_steps=n_steps,
            T=dt * n_steps,
            closed_boundary=True,
            device="cpu",
            use_stable_integrator=True,
            enable_stability_monitor=True,
            rng=np.random.default_rng(42),
        )

        G = sim.backend.copy(sim.G_initial)
        i_over_2 = 0.5 * np.eye(2 * sim.L, dtype=complex)
        sample_steps = []
        distances = []

        for step in range(n_steps):
            G, _ = sim._stable_step_single(G)
            if (step + 1) % 50 == 0:
                g_np = np.asarray(sim.backend.asnumpy(G), dtype=complex)
                sample_steps.append(step + 1)
                distances.append(float(np.linalg.norm(g_np - i_over_2, ord="fro")))

        slope = float(np.polyfit(np.asarray(sample_steps, dtype=float), np.asarray(distances, dtype=float), 1)[0])
        assert slope > -1e-4

    def test_lazy_enforcement_fires_on_threshold(self):
        sim = LQubitCorrelationSimulator(
            L=4,
            J=1.0,
            epsilon=0.01,
            N_steps=4,
            T=0.01,
            closed_boundary=True,
            device="cpu",
            use_stable_integrator=True,
            stable_projector_enforce=False,
            bdg_enforce_threshold=1e-6,
            rng=np.random.default_rng(42),
        )

        calls = {"count": 0}
        original_enforce_bdg = sim._enforce_bdg

        def _counting_enforce_bdg(G):
            calls["count"] += 1
            return original_enforce_bdg(G)

        sim._enforce_bdg = _counting_enforce_bdg  # type: ignore[method-assign]

        g_below = sim._enforce_bdg(sim.backend.copy(sim.G_initial))
        calls_before = calls["count"]
        _ = sim._maybe_enforce_bdg(g_below)
        assert calls["count"] == calls_before

        g_above = sim.backend.copy(sim.G_initial)
        g_above[0, 1] = 0.25 + 0.0j
        calls_before = calls["count"]
        _ = sim._maybe_enforce_bdg(g_above)
        assert calls["count"] == calls_before + 1
