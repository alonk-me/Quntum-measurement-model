"""Pure-Gaussian CPU reference for the continuously monitored Ising chain.

This module is the M2 replacement path for the legacy direct-``G`` integrator.
It fixes the Nambu ordering to ``Psi=(c_1,...,c_L,c_1^dagger,...,c_L^dagger)``
and ``G_ab=<Psi_a^dagger Psi_b>``.  With this ordering the particle-hole swap
is ``tau_x``.  The legacy module's ``tau_y`` identity is not compatible with
that explicit definition.

One split step applies the exact Ising Hamiltonian followed by an exponential
measurement map.  The latter is the frozen-coefficient stochastic exponential
of article Eq. (24):

    exp[sum_x(epsilon*xi_x*M_x/2 - epsilon**2*M_x**2/4)].

Its expansion reproduces Eq. (24) through the declared Ito order.  The same
finite-step map is implemented in Fock space and in the orbital representation
so small systems can be compared on identical standardized Gaussian noise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm


_IDENTITY_2 = np.eye(2, dtype=complex)
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Z = np.diag([1.0, -1.0]).astype(complex)
_FERMION_LOWERING = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _readonly(values: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(values, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def particle_hole_swap(L: int) -> np.ndarray:
    """Return ``tau_x`` for the explicit ``(c,c^dagger)`` Nambu ordering."""

    if L < 1:
        raise ValueError("L must be at least 1")
    zero = np.zeros((L, L), dtype=complex)
    identity = np.eye(L, dtype=complex)
    return np.block([[zero, identity], [identity, zero]])


def fermion_annihilation_operators(L: int) -> list[np.ndarray]:
    """Return Jordan--Wigner annihilators with ``n=(1-Z)/2``."""

    if L < 1:
        raise ValueError("L must be at least 1")
    operators = []
    for site in range(L):
        factors = []
        for index in range(L):
            if index < site:
                factors.append(_PAULI_Z)
            elif index == site:
                factors.append(_FERMION_LOWERING)
            else:
                factors.append(_IDENTITY_2)
        operators.append(_kron_all(factors))
    return operators


def spin_operator(local_operator: np.ndarray, site: int, L: int) -> np.ndarray:
    """Embed a one-site spin operator in an ``L``-site Hilbert space."""

    if not 0 <= site < L:
        raise ValueError("site must satisfy 0 <= site < L")
    return _kron_all(
        [local_operator if index == site else _IDENTITY_2 for index in range(L)]
    )


def _validate_boundary(boundary: str) -> str:
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'")
    return boundary


def ising_spin_hamiltonian(L: int, J: float, boundary: str) -> np.ndarray:
    """Return ``J sum_x X_x X_{x+1}`` in the spin computational basis.

    The periodic definition follows the article's literal directed bond sum.
    Consequently ``L=2`` contains two equal bonds and ``L=1`` contributes only
    the physically irrelevant constant ``J*I``.  Open boundaries use each
    nearest-neighbour bond once.
    """

    if L < 1:
        raise ValueError("L must be at least 1")
    if not np.isfinite(J) or J < 0.0:
        raise ValueError("J must be nonnegative and finite")
    boundary = _validate_boundary(boundary)
    dimension = 2**L
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    if boundary == "open":
        bonds = [(site, site + 1) for site in range(L - 1)]
    else:
        bonds = [(site, (site + 1) % L) for site in range(L)]
    for left, right in bonds:
        if left == right:
            hamiltonian += J * np.eye(dimension, dtype=complex)
        else:
            hamiltonian += J * spin_operator(_PAULI_X, left, L) @ spin_operator(
                _PAULI_X, right, L
            )
    return hamiltonian


def ising_bdg_hamiltonian(
    L: int,
    J: float,
    boundary: str,
    *,
    parity: int,
) -> np.ndarray:
    r"""Return the Hermitian BdG generator in the fixed Nambu convention.

    For an oriented bond ``j -> k``, the Jordan--Wigner image is
    ``J*(c_j^dagger-c_j)*(c_k+c_k^dagger)``.  The periodic boundary bond has
    coefficient ``-parity*J``, where parity is the conserved eigenvalue of
    ``prod_x Z_x=(-1)^N``.  The scalar periodic ``L=1`` term is omitted because
    it has no covariance dynamics.
    """

    if L < 1:
        raise ValueError("L must be at least 1")
    if not np.isfinite(J) or J < 0.0:
        raise ValueError("J must be nonnegative and finite")
    boundary = _validate_boundary(boundary)
    if parity not in (-1, 1):
        raise ValueError("parity must be -1 or +1")

    hopping = np.zeros((L, L), dtype=complex)
    pairing = np.zeros((L, L), dtype=complex)

    def add_oriented_bond(left: int, right: int, coefficient: float) -> None:
        hopping[left, right] += coefficient
        hopping[right, left] += coefficient
        pairing[left, right] += coefficient
        pairing[right, left] -= coefficient

    for site in range(L - 1):
        add_oriented_bond(site, site + 1, J)
    if boundary == "periodic" and L > 1:
        add_oriented_bond(L - 1, 0, -parity * J)

    return np.block(
        [
            [hopping, pairing],
            [-pairing.conj(), -hopping.T],
        ]
    )


def nambu_covariance_from_state(state: np.ndarray, L: int) -> np.ndarray:
    """Construct ``G_ab=<Psi_a^dagger Psi_b>`` from a Fock-space state."""

    state = np.asarray(state, dtype=complex)
    if state.shape != (2**L,):
        raise ValueError(f"state must have shape ({2**L},)")
    norm = np.linalg.norm(state)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("state must be nonzero and finite")
    state = state / norm
    annihilators = fermion_annihilation_operators(L)
    nambu = annihilators + [operator.conj().T for operator in annihilators]
    covariance = np.empty((2 * L, 2 * L), dtype=complex)
    for row, left in enumerate(nambu):
        for column, right in enumerate(nambu):
            covariance[row, column] = np.vdot(
                state, left.conj().T @ right @ state
            )
    return covariance


def article_z_from_covariance(covariance: np.ndarray, L: int) -> np.ndarray:
    """Return article magnetization ``z=1-2<n>`` from the particle block."""

    covariance = np.asarray(covariance, dtype=complex)
    if covariance.shape != (2 * L, 2 * L):
        raise ValueError(f"covariance must have shape ({2 * L}, {2 * L})")
    return 1.0 - 2.0 * np.real(np.diag(covariance)[:L])


def code_z_from_covariance(covariance: np.ndarray, L: int) -> np.ndarray:
    """Return the legacy code magnetization, globally opposite to the article."""

    return -article_z_from_covariance(covariance, L)


def deterministic_phase_qr(
    orbitals: np.ndarray,
    *,
    diagonal_tolerance: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced QR with a positive-real diagonal convention for ``R``.

    Leading dimensions are treated as independent batches.  This lets the CPU
    reference evolve independent trajectories together without changing the
    map applied to any trajectory.
    """

    orbitals = np.asarray(orbitals, dtype=complex)
    if orbitals.ndim < 2 or orbitals.shape[-2] < orbitals.shape[-1]:
        raise ValueError("orbitals must be a tall matrix or batch of tall matrices")
    q_matrix, r_matrix = np.linalg.qr(orbitals, mode="reduced")
    diagonal = np.diagonal(r_matrix, axis1=-2, axis2=-1)
    magnitudes = np.abs(diagonal)
    if np.any(magnitudes <= diagonal_tolerance):
        raise np.linalg.LinAlgError(
            "deterministic QR gauge is ill-conditioned: small R diagonal"
        )
    phases = diagonal / magnitudes
    q_matrix = q_matrix * phases[..., np.newaxis, :]
    r_matrix = phases.conj()[..., :, np.newaxis] * r_matrix
    return q_matrix, r_matrix


def _default_occupied_state(L: int) -> np.ndarray:
    state = np.zeros(2**L, dtype=complex)
    state[-1] = 1.0
    return state


def _normalize_state(state: Optional[np.ndarray], L: int) -> np.ndarray:
    if state is None:
        return _default_occupied_state(L)
    state = np.asarray(state, dtype=complex)
    if state.shape != (2**L,):
        raise ValueError(f"initial_state must have shape ({2**L},)")
    norm = np.linalg.norm(state)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("initial_state must be nonzero and finite")
    return state / norm


def _state_parity(state: np.ndarray, L: int, tolerance: float = 1.0e-12) -> int:
    parity_diagonal = np.array(
        [(-1) ** bin(index).count("1") for index in range(2**L)], dtype=float
    )
    expectation = float(np.vdot(state, parity_diagonal * state).real)
    if abs(abs(expectation) - 1.0) > tolerance:
        raise ValueError("initial_state must have definite fermion parity")
    return 1 if expectation > 0.0 else -1


def _resolve_gaussian_noise(
    *,
    noise: Optional[np.ndarray],
    seed: Optional[int],
    n_trajectories: int,
    n_steps: int,
    L: int,
) -> tuple[np.ndarray, Optional[int]]:
    if (noise is None) == (seed is None):
        raise ValueError("provide exactly one of noise or seed")
    if noise is not None:
        values = np.asarray(noise, dtype=float)
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
        if values.ndim != 3 or values.shape[1:] != (n_steps, L):
            raise ValueError(
                f"noise must have shape ({n_steps}, {L}) or "
                f"(n_trajectories, {n_steps}, {L})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("noise must contain only finite standardized values")
        return values, None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    integer_seed = int(seed)
    generator = np.random.default_rng(integer_seed)
    return generator.standard_normal((n_trajectories, n_steps, L)), integer_seed


def _noise_hash(noise: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(noise)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def coarsen_standard_normal_noise(noise: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a Brownian path while preserving standardized-noise variance.

    If ``dW=sqrt(dt)*xi``, a coarse increment spanning ``factor`` fine steps
    has standardized value ``sum(xi_fine)/sqrt(factor)``.
    """

    noise = np.asarray(noise, dtype=float)
    squeeze = False
    if noise.ndim == 2:
        noise = noise[np.newaxis, :, :]
        squeeze = True
    if noise.ndim != 3 or not np.all(np.isfinite(noise)):
        raise ValueError("noise must be a finite two- or three-dimensional array")
    if isinstance(factor, (bool, np.bool_)) or not isinstance(
        factor, (int, np.integer)
    ):
        raise ValueError("factor must be an integer")
    factor = int(factor)
    if factor < 1 or noise.shape[1] % factor:
        raise ValueError("factor must divide the number of fine noise steps")
    coarsened = noise.reshape(
        noise.shape[0], noise.shape[1] // factor, factor, noise.shape[2]
    ).sum(axis=2) / np.sqrt(factor)
    return coarsened[0] if squeeze else coarsened


def _validate_protocol(
    *,
    L: int,
    gamma: float,
    J: float,
    dt: float,
    n_burnin: int,
    n_samples: int,
    n_trajectories: int,
    boundary: str,
) -> tuple[int, int, int, str]:
    if L < 1:
        raise ValueError("L must be at least 1")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    if not np.isfinite(J) or J < 0.0:
        raise ValueError("J must be nonnegative and finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    for value, name, positive in (
        (n_burnin, "n_burnin", False),
        (n_samples, "n_samples", True),
        (n_trajectories, "n_trajectories", True),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
        if value < int(positive):
            requirement = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be {requirement}")
    return int(n_burnin), int(n_samples), int(n_trajectories), _validate_boundary(boundary)


@dataclass(frozen=True)
class GaussianChainResult:
    """Per-trajectory base-chain observables and algebraic diagnostics."""

    q: np.ndarray
    C0: np.ndarray
    z_history: np.ndarray
    final_covariance: np.ndarray
    max_hermiticity_residual: np.ndarray
    max_particle_hole_residual: np.ndarray
    max_projector_residual: np.ndarray
    max_trace_residual: np.ndarray
    max_eigenvalue_residual: np.ndarray
    max_qr_residual: np.ndarray
    gamma: float
    J: float
    dt: float
    n_burnin: int
    n_samples: int
    boundary: str
    parity: int
    noise_kind: str
    noise_shape: tuple[int, int, int]
    noise_hash: str
    seed: Optional[int]
    integrator: str
    covariance_history: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n_trajectories, n_steps, L = self.noise_shape
        one_dimensional = (
            "q",
            "C0",
            "max_hermiticity_residual",
            "max_particle_hole_residual",
            "max_projector_residual",
            "max_trace_residual",
            "max_eigenvalue_residual",
            "max_qr_residual",
        )
        for name in one_dimensional:
            array = _readonly(getattr(self, name), ndim=1, name=name)
            if array.shape != (n_trajectories,):
                raise ValueError(f"{name} must have shape ({n_trajectories},)")
            object.__setattr__(self, name, array)
        z_history = _readonly(self.z_history, ndim=3, name="z_history")
        if z_history.shape != (n_trajectories, n_steps + 1, L):
            raise ValueError("z_history shape does not match noise metadata")
        object.__setattr__(self, "z_history", z_history)
        final_covariance = _readonly(
            self.final_covariance, ndim=3, name="final_covariance"
        )
        if final_covariance.shape != (n_trajectories, 2 * L, 2 * L):
            raise ValueError("final_covariance shape does not match noise metadata")
        object.__setattr__(self, "final_covariance", final_covariance)
        if self.covariance_history is not None:
            covariance_history = _readonly(
                self.covariance_history, ndim=4, name="covariance_history"
            )
            expected = (n_trajectories, n_steps + 1, 2 * L, 2 * L)
            if covariance_history.shape != expected:
                raise ValueError(
                    "covariance_history shape does not match noise metadata"
                )
            object.__setattr__(self, "covariance_history", covariance_history)

    @property
    def n_trajectories(self) -> int:
        return self.noise_shape[0]

    @property
    def L(self) -> int:
        return self.noise_shape[2]


@dataclass(frozen=True)
class GaussianStationarityDiagnostics:
    """Window and autocorrelation diagnostics for trajectory-resolved ``z**2``.

    Window bounds are indices into ``GaussianChainResult.z_history``: the
    initial state is index zero and index ``k`` is the state after ``k``
    complete split steps.  All reported means retain the trajectory axis so a
    time/site average is never mistaken for an independent-sample error bar.
    """

    window_start_steps: np.ndarray
    window_stop_steps: np.ndarray
    window_mean_z2: np.ndarray
    window_site_mean_z2: np.ndarray
    autocorrelation_time_steps: np.ndarray
    effective_sample_size: np.ndarray
    start_step: int
    stop_step: int
    max_lag: int
    dt: float

    def __post_init__(self) -> None:
        starts = _readonly(self.window_start_steps, ndim=1, name="window_start_steps")
        stops = _readonly(self.window_stop_steps, ndim=1, name="window_stop_steps")
        if starts.shape != stops.shape:
            raise ValueError("window bound arrays must have the same shape")
        object.__setattr__(self, "window_start_steps", starts)
        object.__setattr__(self, "window_stop_steps", stops)

        window_mean = _readonly(self.window_mean_z2, ndim=2, name="window_mean_z2")
        site_mean = _readonly(
            self.window_site_mean_z2, ndim=3, name="window_site_mean_z2"
        )
        if window_mean.shape[1] != starts.size:
            raise ValueError("window_mean_z2 does not match the number of windows")
        if site_mean.shape[:2] != window_mean.shape:
            raise ValueError("window_site_mean_z2 does not match window_mean_z2")
        object.__setattr__(self, "window_mean_z2", window_mean)
        object.__setattr__(self, "window_site_mean_z2", site_mean)

        for name in ("autocorrelation_time_steps", "effective_sample_size"):
            values = _readonly(getattr(self, name), ndim=1, name=name)
            if values.shape != (window_mean.shape[0],):
                raise ValueError(f"{name} does not match the trajectory count")
            object.__setattr__(self, name, values)

    @property
    def autocorrelation_time(self) -> np.ndarray:
        """Integrated autocorrelation time in physical time units."""

        values = self.dt * self.autocorrelation_time_steps
        values.setflags(write=False)
        return values


def _integrated_autocorrelation_time(values: np.ndarray, max_lag: int) -> float:
    """Estimate the integrated autocorrelation time by positive truncation."""

    centered = np.asarray(values, dtype=float) - np.mean(values)
    variance_sum = float(np.dot(centered, centered))
    if variance_sum <= np.finfo(float).eps * values.size:
        return 1.0

    fft_size = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    correlation = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size)
    correlation = correlation[: max_lag + 1] / correlation[0]
    positive = correlation[1 : max_lag + 1]
    nonpositive = np.flatnonzero(positive <= 0.0)
    if nonpositive.size:
        positive = positive[: nonpositive[0]]
    return float(max(1.0, 1.0 + 2.0 * np.sum(positive)))


def chain_stationarity_diagnostics(
    result: GaussianChainResult,
    *,
    start_step: int = 1,
    stop_step: Optional[int] = None,
    n_windows: int = 4,
    max_lag: Optional[int] = None,
) -> GaussianStationarityDiagnostics:
    """Diagnose stationarity without collapsing independent trajectories.

    The scalar series used for autocorrelation is the instantaneous site mean
    ``mean_x(z_x**2)``.  Window diagnostics expose both its site-averaged value
    and the individual-site profiles.  This routine diagnoses a chosen segment;
    it does not decide that the segment is stationary.
    """

    n_history = result.z_history.shape[1]
    if stop_step is None:
        stop_step = n_history
    for value, name in (
        (start_step, "start_step"),
        (stop_step, "stop_step"),
        (n_windows, "n_windows"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
    start_step = int(start_step)
    stop_step = int(stop_step)
    n_windows = int(n_windows)
    if not 0 <= start_step < stop_step <= n_history:
        raise ValueError("require 0 <= start_step < stop_step <= history length")
    n_observations = stop_step - start_step
    if not 1 <= n_windows <= n_observations:
        raise ValueError("n_windows must be between 1 and the segment length")

    if max_lag is None:
        max_lag = min(n_observations - 1, max(1, n_observations // 2))
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(
        max_lag, (int, np.integer)
    ):
        raise ValueError("max_lag must be an integer")
    max_lag = int(max_lag)
    if not 0 <= max_lag < n_observations:
        raise ValueError("max_lag must satisfy 0 <= max_lag < segment length")

    relative_windows = np.array_split(np.arange(n_observations), n_windows)
    starts = np.array(
        [start_step + int(window[0]) for window in relative_windows], dtype=int
    )
    stops = np.array(
        [start_step + int(window[-1]) + 1 for window in relative_windows], dtype=int
    )
    squared = np.square(result.z_history[:, start_step:stop_step, :])
    site_means = np.stack(
        [np.mean(squared[:, window, :], axis=1) for window in relative_windows],
        axis=1,
    )
    window_means = np.mean(site_means, axis=2)
    scalar_series = np.mean(squared, axis=2)
    autocorrelation_times = np.array(
        [
            _integrated_autocorrelation_time(trajectory, max_lag)
            for trajectory in scalar_series
        ]
    )
    effective_sample_size = np.minimum(
        n_observations, n_observations / autocorrelation_times
    )

    return GaussianStationarityDiagnostics(
        window_start_steps=starts,
        window_stop_steps=stops,
        window_mean_z2=window_means,
        window_site_mean_z2=site_means,
        autocorrelation_time_steps=autocorrelation_times,
        effective_sample_size=effective_sample_size,
        start_step=start_step,
        stop_step=stop_step,
        max_lag=max_lag,
        dt=result.dt,
    )


def _basis_z_values(L: int) -> np.ndarray:
    values = np.empty((2**L, L), dtype=float)
    for basis_index in range(2**L):
        for site in range(L):
            bit = (basis_index >> (L - site - 1)) & 1
            values[basis_index, site] = 1.0 - 2.0 * bit
    return values


def _measurement_lambdas(
    z_after_hamiltonian: np.ndarray,
    noise_step: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    return 0.5 * (
        epsilon * noise_step + epsilon * epsilon * z_after_hamiltonian
    )


def _covariance_diagnostics(
    covariance: np.ndarray, tau_x: np.ndarray, L: int
) -> np.ndarray:
    identity = np.eye(2 * L, dtype=complex)
    adjoint = np.swapaxes(covariance.conj(), -2, -1)
    hermiticity = np.linalg.norm(covariance - adjoint, axis=(-2, -1))
    particle_hole = np.linalg.norm(
        tau_x @ covariance.conj() @ tau_x + covariance - identity,
        axis=(-2, -1),
    )
    projector = np.linalg.norm(
        covariance @ covariance - covariance, axis=(-2, -1)
    )
    trace = abs(np.trace(covariance, axis1=-2, axis2=-1).real - L)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + adjoint))
    eigenvalue = np.max(
        np.minimum(abs(eigenvalues), abs(eigenvalues - 1.0)), axis=-1
    )
    return np.stack(
        (hermiticity, particle_hole, projector, trace, eigenvalue), axis=-1
    )


def _make_result(
    *,
    z_history: np.ndarray,
    final_covariance: np.ndarray,
    diagnostics: np.ndarray,
    gamma: float,
    J: float,
    dt: float,
    n_burnin: int,
    n_samples: int,
    boundary: str,
    parity: int,
    noise: np.ndarray,
    seed: Optional[int],
    integrator: str,
    covariance_history: Optional[np.ndarray],
) -> GaussianChainResult:
    retained = z_history[:, n_burnin + 1 :, :]
    mean_z_squared = np.mean(retained * retained, axis=(1, 2))
    return GaussianChainResult(
        q=1.0 + mean_z_squared,
        C0=1.0 - mean_z_squared,
        z_history=z_history,
        final_covariance=final_covariance,
        max_hermiticity_residual=diagnostics[:, 0],
        max_particle_hole_residual=diagnostics[:, 1],
        max_projector_residual=diagnostics[:, 2],
        max_trace_residual=diagnostics[:, 3],
        max_eigenvalue_residual=diagnostics[:, 4],
        max_qr_residual=diagnostics[:, 5],
        gamma=float(gamma),
        J=float(J),
        dt=float(dt),
        n_burnin=n_burnin,
        n_samples=n_samples,
        boundary=boundary,
        parity=parity,
        noise_kind="standard_normal",
        noise_shape=noise.shape,
        noise_hash=_noise_hash(noise),
        seed=seed,
        integrator=integrator,
        covariance_history=covariance_history,
    )


def simulate_exact_small_chain(
    *,
    L: int,
    gamma: float,
    J: float,
    dt: float,
    n_burnin: int,
    n_samples: int,
    boundary: str,
    n_trajectories: int = 1,
    initial_state: Optional[np.ndarray] = None,
    noise: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    store_covariance_history: bool = True,
) -> GaussianChainResult:
    """Run the exact Fock-space reference with the M2 exponential split map."""

    n_burnin, n_samples, n_trajectories, boundary = _validate_protocol(
        L=L,
        gamma=gamma,
        J=J,
        dt=dt,
        n_burnin=n_burnin,
        n_samples=n_samples,
        n_trajectories=n_trajectories,
        boundary=boundary,
    )
    state0 = _normalize_state(initial_state, L)
    parity = _state_parity(state0, L)
    n_steps = n_burnin + n_samples
    noise, recorded_seed = _resolve_gaussian_noise(
        noise=noise,
        seed=seed,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        L=L,
    )
    n_trajectories = noise.shape[0]
    states = np.broadcast_to(state0, (n_trajectories, 2**L)).copy()
    unitary = expm(-1.0j * ising_spin_hamiltonian(L, J, boundary) * dt)
    z_basis = _basis_z_values(L)
    tau_x = particle_hole_swap(L)
    epsilon = np.sqrt(gamma * dt)

    z_history = np.empty((n_trajectories, n_steps + 1, L), dtype=float)
    final_covariance = np.empty((n_trajectories, 2 * L, 2 * L), dtype=complex)
    covariance_history = None
    if store_covariance_history:
        covariance_history = np.empty(
            (n_trajectories, n_steps + 1, 2 * L, 2 * L), dtype=complex
        )
    diagnostics = np.zeros((n_trajectories, 6), dtype=float)

    for trajectory in range(n_trajectories):
        covariance = nambu_covariance_from_state(states[trajectory], L)
        z_history[trajectory, 0] = article_z_from_covariance(covariance, L)
        if covariance_history is not None:
            covariance_history[trajectory, 0] = covariance
        diagnostics[trajectory, :5] = _covariance_diagnostics(
            covariance, tau_x, L
        )

    for step in range(n_steps):
        for trajectory in range(n_trajectories):
            state_h = unitary @ states[trajectory]
            covariance_h = nambu_covariance_from_state(state_h, L)
            z_h = article_z_from_covariance(covariance_h, L)
            lambdas = _measurement_lambdas(
                z_h, noise[trajectory, step], epsilon
            )
            amplitudes = np.exp(z_basis @ lambdas)
            state = amplitudes * state_h
            state_norm = np.linalg.norm(state)
            if state_norm == 0.0 or not np.isfinite(state_norm):
                raise FloatingPointError("measurement map produced invalid norm")
            states[trajectory] = state / state_norm
            covariance = nambu_covariance_from_state(states[trajectory], L)
            z_history[trajectory, step + 1] = article_z_from_covariance(
                covariance, L
            )
            if covariance_history is not None:
                covariance_history[trajectory, step + 1] = covariance
            current = np.array(_covariance_diagnostics(covariance, tau_x, L))
            diagnostics[trajectory, :5] = np.maximum(
                diagnostics[trajectory, :5], current
            )
            final_covariance[trajectory] = covariance

    return _make_result(
        z_history=z_history,
        final_covariance=final_covariance,
        diagnostics=diagnostics,
        gamma=gamma,
        J=J,
        dt=dt,
        n_burnin=n_burnin,
        n_samples=n_samples,
        boundary=boundary,
        parity=parity,
        noise=noise,
        seed=recorded_seed,
        integrator="exact_fock_exponential_split_v1",
        covariance_history=covariance_history,
    )


def simulate_gaussian_orbital_chain(
    *,
    L: int,
    gamma: float,
    J: float,
    dt: float,
    n_burnin: int,
    n_samples: int,
    boundary: str,
    n_trajectories: int = 1,
    initial_state: Optional[np.ndarray] = None,
    noise: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    store_covariance_history: bool = False,
) -> GaussianChainResult:
    """Run the CPU purity-preserving orbital/QR base chain without tangents."""

    n_burnin, n_samples, n_trajectories, boundary = _validate_protocol(
        L=L,
        gamma=gamma,
        J=J,
        dt=dt,
        n_burnin=n_burnin,
        n_samples=n_samples,
        n_trajectories=n_trajectories,
        boundary=boundary,
    )
    state0 = _normalize_state(initial_state, L)
    parity = _state_parity(state0, L)
    covariance0 = nambu_covariance_from_state(state0, L)
    gaussian_residual = np.linalg.norm(
        covariance0 @ covariance0 - covariance0, ord="fro"
    )
    if gaussian_residual > 1.0e-10:
        raise ValueError("initial_state is not a pure Gaussian state")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance0)
    occupied = eigenvectors[:, eigenvalues > 0.5]
    if occupied.shape != (2 * L, L):
        raise ValueError("initial covariance does not have rank L")
    occupied, _ = deterministic_phase_qr(occupied)

    n_steps = n_burnin + n_samples
    noise, recorded_seed = _resolve_gaussian_noise(
        noise=noise,
        seed=seed,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        L=L,
    )
    n_trajectories = noise.shape[0]
    orbitals = np.broadcast_to(occupied, (n_trajectories, 2 * L, L)).copy()
    bdg_hamiltonian = ising_bdg_hamiltonian(
        L, J, boundary, parity=parity
    )
    orbital_hamiltonian_map = expm(1.0j * bdg_hamiltonian * dt)
    tau_x = particle_hole_swap(L)
    epsilon = np.sqrt(gamma * dt)

    z_history = np.empty((n_trajectories, n_steps + 1, L), dtype=float)
    final_covariance = np.empty((n_trajectories, 2 * L, 2 * L), dtype=complex)
    covariance_history = None
    if store_covariance_history:
        covariance_history = np.empty(
            (n_trajectories, n_steps + 1, 2 * L, 2 * L), dtype=complex
        )
    diagnostics = np.zeros((n_trajectories, 6), dtype=float)

    covariance = orbitals @ np.swapaxes(orbitals.conj(), -2, -1)
    z_history[:, 0, :] = 1.0 - 2.0 * np.real(
        np.diagonal(covariance, axis1=-2, axis2=-1)[:, :L]
    )
    if covariance_history is not None:
        covariance_history[:, 0, :, :] = covariance
    diagnostics[:, :5] = _covariance_diagnostics(covariance, tau_x, L)

    for step in range(n_steps):
        after_hamiltonian = orbital_hamiltonian_map @ orbitals
        covariance_h = after_hamiltonian @ np.swapaxes(
            after_hamiltonian.conj(), -2, -1
        )
        z_h = 1.0 - 2.0 * np.real(
            np.diagonal(covariance_h, axis1=-2, axis2=-1)[:, :L]
        )
        lambdas = _measurement_lambdas(z_h, noise[:, step, :], epsilon)
        diagonal_map = np.concatenate(
            (np.exp(-2.0 * lambdas), np.exp(2.0 * lambdas)), axis=1
        )
        raw_orbitals = diagonal_map[:, :, np.newaxis] * after_hamiltonian
        orbitals, _ = deterministic_phase_qr(raw_orbitals)
        qr_residual = np.linalg.norm(
            np.swapaxes(orbitals.conj(), -2, -1) @ orbitals
            - np.eye(L),
            axis=(-2, -1),
        )
        covariance = orbitals @ np.swapaxes(orbitals.conj(), -2, -1)
        z_history[:, step + 1, :] = 1.0 - 2.0 * np.real(
            np.diagonal(covariance, axis1=-2, axis2=-1)[:, :L]
        )
        if covariance_history is not None:
            covariance_history[:, step + 1, :, :] = covariance
        current = _covariance_diagnostics(covariance, tau_x, L)
        diagnostics[:, :5] = np.maximum(diagnostics[:, :5], current)
        diagnostics[:, 5] = np.maximum(diagnostics[:, 5], qr_residual)

    final_covariance[:, :, :] = covariance

    return _make_result(
        z_history=z_history,
        final_covariance=final_covariance,
        diagnostics=diagnostics,
        gamma=gamma,
        J=J,
        dt=dt,
        n_burnin=n_burnin,
        n_samples=n_samples,
        boundary=boundary,
        parity=parity,
        noise=noise,
        seed=recorded_seed,
        integrator="pure_gaussian_orbital_exponential_split_v1",
        covariance_history=covariance_history,
    )
