#!/usr/bin/env python3
"""Select a small-chain burn-in and sampling protocol from stationarity data.

The run evolves one long, explicitly seeded Gaussian-noise history per ``g``
value.  Nominal, extended-burn-in, extended-duration, and fully extended
segments are then compared trajectory by trajectory.  A protocol is recommended
only when both extension tests, the first-to-last retained-window test, and a
coupled-Brownian half-timestep test change the ensemble mean by less than its
independent-trajectory 95% uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Optional

import numpy as np
import scipy

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from quantum_measurement.jw_expansion.gaussian_orbital import (
    GaussianChainResult,
    chain_stationarity_diagnostics,
    coarsen_standard_normal_noise,
    simulate_gaussian_orbital_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g-values", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--burn-in-time", type=float, default=5.0)
    parser.add_argument("--sample-time", type=float, default=10.0)
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--boundary", choices=("open", "periodic"), default="periodic")
    parser.add_argument("--n-windows", type=int, default=4)
    parser.add_argument(
        "--dt-refinement-factor",
        type=int,
        default=2,
        help="Compare dt against dt/factor on a coupled Brownian path; use 1 to disable.",
    )
    parser.add_argument(
        "--max-lag-time",
        type=float,
        default=None,
        help="Autocorrelation cutoff in physical time; defaults to half a segment.",
    )
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_gaussian_orbital_stationarity"),
    )
    return parser.parse_args()


def _step_count(duration: float, dt: float, name: str) -> int:
    steps = int(round(duration / dt))
    if steps <= 0 or not np.isclose(steps * dt, duration, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return steps


def _validate_arguments(args: argparse.Namespace) -> tuple[int, int, Optional[int]]:
    if args.L < 1:
        raise ValueError("L must be positive")
    if args.J <= 0.0 or not np.isfinite(args.J):
        raise ValueError("J must be positive and finite so g=gamma/(4J) is defined")
    if args.dt <= 0.0 or not np.isfinite(args.dt):
        raise ValueError("dt must be positive and finite")
    if any(value <= 0.0 or not np.isfinite(value) for value in args.g_values):
        raise ValueError("all g-values must be positive and finite")
    if args.n_trajectories < 2:
        raise ValueError("at least two independent trajectories are required")
    if args.n_windows < 2:
        raise ValueError("at least two stationarity windows are required")
    if args.dt_refinement_factor < 1:
        raise ValueError("dt-refinement-factor must be a positive integer")
    burn_in_steps = _step_count(args.burn_in_time, args.dt, "burn-in-time")
    sample_steps = _step_count(args.sample_time, args.dt, "sample-time")
    if args.n_windows > sample_steps:
        raise ValueError("n-windows cannot exceed the nominal sample count")
    max_lag = None
    if args.max_lag_time is not None:
        max_lag = _step_count(args.max_lag_time, args.dt, "max-lag-time")
        if max_lag >= sample_steps:
            raise ValueError("max-lag-time must be shorter than sample-time")
    return burn_in_steps, sample_steps, max_lag


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    mean, sem = _mean_sem(values)
    return {
        "values": values.tolist(),
        "mean": mean,
        "std": float(np.std(values, ddof=1)),
        "sem": sem,
        "confidence_95_half_width": 1.96 * sem,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0.05, 0.5, 0.95), np.quantile(values, (0.05, 0.5, 0.95))
            )
        },
    }


def _segment_q(
    result: GaussianChainResult, *, start_after_steps: int, n_steps: int
) -> np.ndarray:
    start = start_after_steps + 1
    stop = start + n_steps
    z = result.z_history[:, start:stop, :]
    if z.shape[1] != n_steps:
        raise ValueError("requested segment extends beyond the stored history")
    return 1.0 + np.mean(np.square(z), axis=(1, 2))


def _extension_comparison(reference: np.ndarray, extended: np.ndarray) -> dict[str, object]:
    reference_mean, reference_sem = _mean_sem(reference)
    extended_mean, extended_sem = _mean_sem(extended)
    difference = np.asarray(extended) - np.asarray(reference)
    difference_mean, difference_sem = _mean_sem(difference)
    uncertainty_limit = max(1.96 * reference_sem, 1.96 * extended_sem)
    passed = abs(difference_mean) <= uncertainty_limit + 32.0 * np.finfo(float).eps
    return {
        "reference_mean": reference_mean,
        "reference_sem": reference_sem,
        "extended_mean": extended_mean,
        "extended_sem": extended_sem,
        "mean_shift": difference_mean,
        "paired_shift_sem": difference_sem,
        "paired_shift_confidence_95_half_width": 1.96 * difference_sem,
        "acceptance_uncertainty": uncertainty_limit,
        "passed": bool(passed),
    }


def _git_metadata() -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            arguments,
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        dirty = bool(run("git", "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _stationarity_record(
    result: GaussianChainResult,
    *,
    burn_in_steps: int,
    sample_steps: int,
    n_windows: int,
    max_lag: Optional[int],
) -> dict[str, object]:
    segments = {
        "nominal": _segment_q(
            result, start_after_steps=burn_in_steps, n_steps=sample_steps
        ),
        "extended_burn_in": _segment_q(
            result, start_after_steps=2 * burn_in_steps, n_steps=sample_steps
        ),
        "extended_duration": _segment_q(
            result, start_after_steps=burn_in_steps, n_steps=2 * sample_steps
        ),
        "fully_extended": _segment_q(
            result, start_after_steps=2 * burn_in_steps, n_steps=2 * sample_steps
        ),
    }
    diagnostics = chain_stationarity_diagnostics(
        result,
        start_step=burn_in_steps + 1,
        stop_step=burn_in_steps + sample_steps + 1,
        n_windows=n_windows,
        max_lag=max_lag,
    )
    first_window = diagnostics.window_mean_z2[:, 0]
    last_window = diagnostics.window_mean_z2[:, -1]
    window_comparison = _extension_comparison(first_window, last_window)
    local_shift = (
        diagnostics.window_site_mean_z2[:, -1, :]
        - diagnostics.window_site_mean_z2[:, 0, :]
    )
    local_shift_mean = np.mean(local_shift, axis=0)
    local_shift_sem = np.std(local_shift, axis=0, ddof=1) / np.sqrt(
        local_shift.shape[0]
    )

    burn_in_comparison = _extension_comparison(
        segments["nominal"], segments["extended_burn_in"]
    )
    duration_comparison = _extension_comparison(
        segments["nominal"], segments["extended_duration"]
    )
    accepted = (
        burn_in_comparison["passed"]
        and duration_comparison["passed"]
        and window_comparison["passed"]
    )
    return {
        "segments": {name: _distribution(values) for name, values in segments.items()},
        "comparisons": {
            "extended_burn_in": burn_in_comparison,
            "extended_duration": duration_comparison,
            "first_to_last_window_z2": window_comparison,
        },
        "windows": {
            "start_steps": diagnostics.window_start_steps.tolist(),
            "stop_steps": diagnostics.window_stop_steps.tolist(),
            "ensemble_mean_z2": np.mean(diagnostics.window_mean_z2, axis=0).tolist(),
            "ensemble_sem_z2": (
                np.std(diagnostics.window_mean_z2, axis=0, ddof=1)
                / np.sqrt(result.n_trajectories)
            ).tolist(),
            "ensemble_site_mean_z2": np.mean(
                diagnostics.window_site_mean_z2, axis=0
            ).tolist(),
            "first_to_last_local_shift": local_shift_mean.tolist(),
            "first_to_last_local_shift_sem": local_shift_sem.tolist(),
        },
        "autocorrelation": {
            "max_lag_steps": diagnostics.max_lag,
            "time_steps": _distribution(diagnostics.autocorrelation_time_steps),
            "physical_time": _distribution(diagnostics.autocorrelation_time),
            "effective_sample_size": _distribution(diagnostics.effective_sample_size),
        },
        "accepted": bool(accepted),
    }


def main() -> None:
    args = parse_args()
    burn_in_steps, sample_steps, max_lag = _validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 2 * burn_in_steps + 2 * sample_steps
    records = []
    rows = []

    for index, g_value in enumerate(args.g_values):
        gamma = 4.0 * args.J * g_value
        seed = args.seed + index
        start = perf_counter()
        result = simulate_gaussian_orbital_chain(
            L=args.L,
            gamma=gamma,
            J=args.J,
            dt=args.dt,
            n_burnin=0,
            n_samples=total_steps,
            boundary=args.boundary,
            n_trajectories=args.n_trajectories,
            seed=seed,
        )
        elapsed = perf_counter() - start
        stationarity = _stationarity_record(
            result,
            burn_in_steps=burn_in_steps,
            sample_steps=sample_steps,
            n_windows=args.n_windows,
            max_lag=max_lag,
        )
        timestep_refinement = None
        if args.dt_refinement_factor > 1:
            factor = args.dt_refinement_factor
            fine_dt = args.dt / factor
            fine_burn_in_steps = burn_in_steps * factor
            fine_sample_steps = sample_steps * factor
            fine_steps = fine_burn_in_steps + fine_sample_steps
            fine_generator = np.random.default_rng(args.seed + 10000 + index)
            fine_noise = fine_generator.standard_normal(
                (args.n_trajectories, fine_steps, args.L)
            )
            coarse_noise = coarsen_standard_normal_noise(fine_noise, factor)
            coarse = simulate_gaussian_orbital_chain(
                L=args.L,
                gamma=gamma,
                J=args.J,
                dt=args.dt,
                n_burnin=burn_in_steps,
                n_samples=sample_steps,
                boundary=args.boundary,
                noise=coarse_noise,
            )
            fine = simulate_gaussian_orbital_chain(
                L=args.L,
                gamma=gamma,
                J=args.J,
                dt=fine_dt,
                n_burnin=fine_burn_in_steps,
                n_samples=fine_sample_steps,
                boundary=args.boundary,
                noise=fine_noise,
            )
            timestep_refinement = _extension_comparison(coarse.q, fine.q)
            timestep_refinement.update(
                {
                    "coarse_dt": args.dt,
                    "fine_dt": fine_dt,
                    "factor": factor,
                    "coarse_noise_hash": coarse.noise_hash,
                    "fine_noise_hash": fine.noise_hash,
                    "coupling": "coarse xi=sum(fine xi)/sqrt(factor)",
                }
            )
            stationarity["accepted"] = bool(
                stationarity["accepted"] and timestep_refinement["passed"]
            )
        invariant_maxima = {
            name: float(np.max(getattr(result, name)))
            for name in (
                "max_hermiticity_residual",
                "max_particle_hole_residual",
                "max_projector_residual",
                "max_trace_residual",
                "max_eigenvalue_residual",
                "max_qr_residual",
            )
        }
        record = {
            "g": g_value,
            "gamma": gamma,
            "seed": seed,
            "noise_hash": result.noise_hash,
            "runtime_seconds": elapsed,
            "stationarity": stationarity,
            "timestep_refinement": timestep_refinement,
            "invariant_maxima": invariant_maxima,
        }
        records.append(record)
        nominal = stationarity["segments"]["nominal"]
        burn_in = stationarity["comparisons"]["extended_burn_in"]
        duration = stationarity["comparisons"]["extended_duration"]
        windows = stationarity["comparisons"]["first_to_last_window_z2"]
        rows.append(
            {
                "g": g_value,
                "gamma": gamma,
                "q_mean": nominal["mean"],
                "q_sem": nominal["sem"],
                "burn_in_extension_shift": burn_in["mean_shift"],
                "burn_in_extension_passed": burn_in["passed"],
                "duration_extension_shift": duration["mean_shift"],
                "duration_extension_passed": duration["passed"],
                "window_shift": windows["mean_shift"],
                "window_test_passed": windows["passed"],
                "timestep_refinement_shift": (
                    timestep_refinement["mean_shift"]
                    if timestep_refinement is not None
                    else None
                ),
                "timestep_refinement_passed": (
                    timestep_refinement["passed"]
                    if timestep_refinement is not None
                    else None
                ),
                "stationarity_accepted": stationarity["accepted"],
                "median_autocorrelation_time": stationarity["autocorrelation"][
                    "physical_time"
                ]["quantiles"]["0.5"],
                "runtime_seconds": elapsed,
            }
        )
        print(
            f"g={g_value:g}: q={nominal['mean']:.8f} +/- {nominal['sem']:.2g}; "
            f"burn-in={burn_in['passed']}, duration={duration['passed']}, "
            f"windows={windows['passed']}, "
            f"dt={timestep_refinement['passed'] if timestep_refinement else 'not-run'} "
            f"({elapsed:.1f} s base run)"
        )

    accepted = all(record["stationarity"]["accepted"] for record in records)
    configuration = {
        "L": args.L,
        "g_values": args.g_values,
        "J": args.J,
        "dt": args.dt,
        "burn_in_time": args.burn_in_time,
        "sample_time": args.sample_time,
        "n_trajectories": args.n_trajectories,
        "boundary": args.boundary,
        "n_windows": args.n_windows,
        "dt_refinement_factor": args.dt_refinement_factor,
        "max_lag_time": args.max_lag_time,
        "seed": args.seed,
        "initial_state": "computational |1>^L (article z_x=-1)",
        "noise_kind": "standard_normal",
        "gamma_relation": "gamma=4*J*g",
        "segment_policy": {
            "nominal": "burn_in_time followed by sample_time",
            "extended_burn_in": "2*burn_in_time followed by sample_time",
            "extended_duration": "burn_in_time followed by 2*sample_time",
            "fully_extended": "2*burn_in_time followed by 2*sample_time",
        },
        "acceptance_rule": (
            "burn-in, retained-duration, and first-to-last-window ensemble-mean "
            "shifts, plus the coupled-Brownian timestep-refinement shift, must "
            "each be no larger than the larger independent-trajectory 95% "
            "confidence half-width of the two estimates"
        ),
    }
    payload = {
        "schema_version": 1,
        "configuration": configuration,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git": _git_metadata(),
        },
        "protocol_recommended": accepted,
        "results": records,
    }
    json_path = args.output_dir / "stationarity_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output_dir / "stationarity_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"protocol_recommended={accepted}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
