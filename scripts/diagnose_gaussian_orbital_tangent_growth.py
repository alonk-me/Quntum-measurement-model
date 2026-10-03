#!/usr/bin/env python3
"""Diagnose long-time Gaussian-orbital tangent growth without repairing it.

The primary run replays the frozen L=8 path.  Common-noise finite differences
are short diagnostic references only.  A long L=2 exact-Fock control separates
orbital-coordinate effects from sensitivity of the physical split map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantum_measurement.jw_expansion.gaussian_orbital import (
    simulate_exact_small_chain,
    simulate_gaussian_orbital_chain,
    simulate_gaussian_orbital_chain_second_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--burn-in-time", type=float, default=5.0)
    parser.add_argument("--sample-time", type=float, default=10.0)
    parser.add_argument("--n-trajectories", type=int, default=8)
    parser.add_argument("--boundary", choices=("open", "periodic"), default="periodic")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--h-values", nargs="+", type=float, default=[0.02, 0.01, 0.005, 0.0025]
    )
    parser.add_argument("--fd-trajectories", type=int, default=2)
    parser.add_argument("--exact-control-trajectories", type=int, default=2)
    parser.add_argument("--record-stride", type=int, default=10)
    parser.add_argument("--skip-finite-difference", action="store_true")
    parser.add_argument("--skip-exact-control", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_gaussian_orbital_tangent_growth"),
    )
    return parser.parse_args()


def _step_count(duration: float, dt: float, name: str) -> int:
    count = int(round(duration / dt))
    if count < 0 or not np.isclose(count * dt, duration, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a nonnegative integer multiple of dt")
    return count


def _validate(args: argparse.Namespace) -> tuple[int, int]:
    if args.L < 2 or args.g <= 0.0 or args.J <= 0.0 or args.dt <= 0.0:
        raise ValueError("require L >= 2 and positive g, J, and dt")
    if args.n_trajectories < 2:
        raise ValueError("n-trajectories must be at least two")
    if not 1 <= args.fd_trajectories <= args.n_trajectories:
        raise ValueError("fd-trajectories must lie between one and n-trajectories")
    if not 1 <= args.exact_control_trajectories <= args.n_trajectories:
        raise ValueError(
            "exact-control-trajectories must lie between one and n-trajectories"
        )
    if not args.h_values or any(h <= 0.0 for h in args.h_values):
        raise ValueError("all h-values must be positive")
    if args.record_stride < 1:
        raise ValueError("record-stride must be positive")
    burn_in = _step_count(args.burn_in_time, args.dt, "burn-in-time")
    samples = _step_count(args.sample_time, args.dt, "sample-time")
    if samples < 1:
        raise ValueError("sample-time must include at least one step")
    return burn_in, samples


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    ddof = 1 if values.size > 1 else 0
    sem = float(np.std(values, ddof=ddof) / np.sqrt(values.size))
    return {
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=ddof)),
        "sem": sem,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "median": float(np.median(values)),
    }


def _peak(values: np.ndarray, dt: float) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    steps = np.argmax(values, axis=1)
    maxima = values[np.arange(values.shape[0]), steps]
    return {
        "maximum": _distribution(maxima),
        "step": steps.tolist(),
        "time": (steps * dt).tolist(),
    }


def _threshold_times(values: np.ndarray, dt: float) -> dict[str, list[object]]:
    output = {}
    for threshold in (1.0e2, 1.0e4, 1.0e6, 1.0e8):
        crossings = []
        for trajectory in values:
            indices = np.flatnonzero(trajectory >= threshold)
            crossings.append(None if not indices.size else float(indices[0] * dt))
        output[f"{threshold:.0e}"] = crossings
    return output


def _growth_summary(result, history, dt: float, n_burnin: int) -> dict[str, object]:
    growth = history.growth
    if growth is None:
        raise RuntimeError("tangent-growth history was not recorded")
    names = (
        "first_orbital_norm",
        "second_orbital_norm",
        "first_covariance_norm",
        "second_covariance_norm",
        "first_within_occupied_norm",
        "first_horizontal_norm",
        "second_within_occupied_norm",
        "second_horizontal_norm",
        "second_covariance_cancellation_ratio",
    )
    peaks = {name: _peak(getattr(growth, name), dt) for name in names}
    max_F2 = np.max(growth.second_covariance_norm, axis=1)
    max_V2 = np.max(growth.second_orbital_norm, axis=1)
    retained = slice(n_burnin + 1, None)
    positive = np.mean(2.0 * np.square(history.u[:, retained, :]), axis=(1, 2))
    signed = np.mean(
        2.0 * history.z[:, retained, :] * history.v[:, retained, :], axis=(1, 2)
    )
    return {
        "peaks": peaks,
        "second_orbital_to_covariance_peak_ratio": _distribution(
            max_V2 / np.maximum(max_F2, np.finfo(float).tiny)
        ),
        "second_orbital_threshold_times": _threshold_times(
            growth.second_orbital_norm, dt
        ),
        "second_covariance_threshold_times": _threshold_times(
            growth.second_covariance_norm, dt
        ),
        "minimum_qr_diagonal": _distribution(
            np.min(growth.minimum_qr_diagonal, axis=1)
        ),
        "maximum_invariant_residual_by_kind_and_order": np.max(
            growth.invariant_residuals, axis=(0, 1)
        ).tolist(),
        "q": _distribution(result.q),
        "dq": _distribution(result.dq_dtheta),
        "d2q": _distribution(result.d2q_dtheta2),
        "d2q_positive_2u2": _distribution(positive),
        "d2q_signed_2zv": _distribution(signed),
        "d2q_reconstruction_error": float(
            np.max(np.abs(result.d2q_dtheta2 - positive - signed))
        ),
    }


def _agreement_horizon(
    direct: np.ndarray, reference: np.ndarray, dt: float, threshold: float = 0.1
) -> list[object]:
    direct_norm = np.linalg.norm(direct, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    error = np.linalg.norm(direct - reference, axis=-1)
    scale = np.maximum(1.0, np.maximum(direct_norm, reference_norm))
    normalized = error / scale
    output = []
    for trajectory in normalized:
        indices = np.flatnonzero(trajectory > threshold)
        output.append(None if not indices.size else float(indices[0] * dt))
    return output


def _finite_difference_sweep(
    *,
    runner,
    common: dict[str, object],
    noise: np.ndarray,
    tangent,
    history,
    central_q: np.ndarray,
    central_covariance: np.ndarray,
    h_values: list[float],
    n_burnin: int,
) -> dict[str, object]:
    records = []
    estimates = []
    retained = slice(n_burnin + 1, None)
    for h in h_values:
        start = perf_counter()
        plus = runner(
            **dict(common, gamma=common["gamma"] * np.exp(h)), noise=noise
        )
        minus = runner(
            **dict(common, gamma=common["gamma"] * np.exp(-h)), noise=noise
        )
        finite_u = (plus.z_history - minus.z_history) / (2.0 * h)
        finite_v = (
            plus.z_history - 2.0 * history.z + minus.z_history
        ) / (h * h)
        finite_dq = (plus.q - minus.q) / (2.0 * h)
        finite_d2q = (plus.q - 2.0 * central_q + minus.q) / (h * h)
        finite_F = (
            plus.final_covariance - minus.final_covariance
        ) / (2.0 * h)
        finite_F2 = (
            plus.final_covariance
            - 2.0 * central_covariance
            + minus.final_covariance
        ) / (h * h)
        estimates.append((finite_v, finite_d2q, finite_F2))
        records.append(
            {
                "h": h,
                "seconds": perf_counter() - start,
                "u_rms_error": _rms(history.u - finite_u),
                "v_rms_error": _rms(history.v - finite_v),
                "retained_v_rms_error": _rms(
                    history.v[:, retained, :] - finite_v[:, retained, :]
                ),
                "dq_rms_error": _rms(tangent.dq_dtheta - finite_dq),
                "d2q_rms_error": _rms(tangent.d2q_dtheta2 - finite_d2q),
                "F_rms_error": _rms(tangent.final_first_covariance - finite_F),
                "F2_rms_error": _rms(
                    tangent.final_second_covariance - finite_F2
                ),
                "u_agreement_horizon_10pct": _agreement_horizon(
                    history.u, finite_u, common["dt"]
                ),
                "v_agreement_horizon_10pct": _agreement_horizon(
                    history.v, finite_v, common["dt"]
                ),
                "finite_d2q": finite_d2q.tolist(),
                "direct_d2q": tangent.d2q_dtheta.tolist(),
                "finite_F2_norm": np.linalg.norm(
                    finite_F2, axis=(-2, -1)
                ).tolist(),
                "direct_F2_norm": np.linalg.norm(
                    tangent.final_second_covariance, axis=(-2, -1)
                ).tolist(),
            }
        )
    successive = []
    for index in range(len(estimates) - 1):
        coarse_v, coarse_d2q, coarse_F2 = estimates[index]
        fine_v, fine_d2q, fine_F2 = estimates[index + 1]
        successive.append(
            {
                "coarse_h": h_values[index],
                "fine_h": h_values[index + 1],
                "v_estimate_rms_difference": _rms(coarse_v - fine_v),
                "d2q_estimate_rms_difference": _rms(coarse_d2q - fine_d2q),
                "F2_estimate_rms_difference": _rms(coarse_F2 - fine_F2),
            }
        )
    return {"h_records": records, "successive_estimate_differences": successive}


def _select_trajectories(history, count: int) -> list[int]:
    physical_peaks = np.max(history.growth.second_covariance_norm, axis=1)
    orbital_peaks = np.max(history.growth.second_orbital_norm, axis=1)
    order = []
    for candidate in np.argsort(physical_peaks)[::-1]:
        if int(candidate) not in order:
            order.append(int(candidate))
    for candidate in np.argsort(orbital_peaks)[::-1]:
        if int(candidate) not in order:
            order.append(int(candidate))
    return order[:count]


def _write_timeseries(
    path: Path, history, dt: float, stride: int, label: str
) -> None:
    growth = history.growth
    steps = list(range(0, history.z.shape[1], stride))
    if steps[-1] != history.z.shape[1] - 1:
        steps.append(history.z.shape[1] - 1)
    metric_names = (
        "first_orbital_norm",
        "second_orbital_norm",
        "first_covariance_norm",
        "second_covariance_norm",
        "first_within_occupied_norm",
        "first_horizontal_norm",
        "second_within_occupied_norm",
        "second_horizontal_norm",
        "second_covariance_linear_norm",
        "second_covariance_quadratic_norm",
        "second_covariance_cancellation_ratio",
        "minimum_qr_diagonal",
    )
    fields = [
        "run",
        "trajectory",
        "step",
        "time",
        "max_abs_z",
        "max_abs_u",
        "max_abs_v",
        "instantaneous_2u2",
        "instantaneous_2zv",
        *metric_names,
        "max_state_invariant",
        "max_first_invariant",
        "max_second_invariant",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trajectory in range(history.z.shape[0]):
            for step in steps:
                row = {
                    "run": label,
                    "trajectory": trajectory,
                    "step": step,
                    "time": step * dt,
                    "max_abs_z": np.max(np.abs(history.z[trajectory, step])),
                    "max_abs_u": np.max(np.abs(history.u[trajectory, step])),
                    "max_abs_v": np.max(np.abs(history.v[trajectory, step])),
                    "instantaneous_2u2": 2.0
                    * np.mean(np.square(history.u[trajectory, step])),
                    "instantaneous_2zv": 2.0
                    * np.mean(
                        history.z[trajectory, step] * history.v[trajectory, step]
                    ),
                    "max_state_invariant": np.max(
                        growth.invariant_residuals[trajectory, step, :, 0]
                    ),
                    "max_first_invariant": np.max(
                        growth.invariant_residuals[trajectory, step, :, 1]
                    ),
                    "max_second_invariant": np.max(
                        growth.invariant_residuals[trajectory, step, :, 2]
                    ),
                }
                row.update(
                    {
                        name: getattr(growth, name)[trajectory, step]
                        for name in metric_names
                    }
                )
                writer.writerow(row)


def _git_metadata() -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            arguments, cwd=_ROOT, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    n_burnin, n_samples = _validate(args)
    n_steps = n_burnin + n_samples
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gamma = 4.0 * args.J * args.g
    noise = np.random.default_rng(args.seed).standard_normal(
        (args.n_trajectories, n_steps, args.L)
    )
    common = {
        "L": args.L,
        "gamma": gamma,
        "J": args.J,
        "dt": args.dt,
        "n_burnin": n_burnin,
        "n_samples": n_samples,
        "boundary": args.boundary,
    }

    start = perf_counter()
    tangent, history = simulate_gaussian_orbital_chain_second_order(
        **common, noise=noise, store_history=True
    )
    primary_seconds = perf_counter() - start
    selected = _select_trajectories(history, args.fd_trajectories)
    primary_summary = _growth_summary(tangent, history, args.dt, n_burnin)
    primary_summary["selected_finite_difference_trajectories"] = selected
    primary_summary["seconds"] = primary_seconds
    print(
        "primary: max(V2,F2)="
        f"({np.max(history.growth.second_orbital_norm):.3e}, "
        f"{np.max(history.growth.second_covariance_norm):.3e}); "
        f"selected={selected}"
    )

    finite_difference = None
    if not args.skip_finite_difference:
        selected_noise = noise[selected]
        selected_tangent, selected_history = (
            simulate_gaussian_orbital_chain_second_order(
                **common, noise=selected_noise, store_history=True
            )
        )
        finite_difference = _finite_difference_sweep(
            runner=simulate_gaussian_orbital_chain,
            common=common,
            noise=selected_noise,
            tangent=selected_tangent,
            history=selected_history,
            central_q=selected_tangent.q,
            central_covariance=selected_tangent.final_covariance,
            h_values=args.h_values,
            n_burnin=n_burnin,
        )

    exact_control = None
    control_history = None
    if not args.skip_exact_control:
        count = args.exact_control_trajectories
        control_noise = noise[:count, :, :2]
        control_common = dict(common, L=2)
        start = perf_counter()
        control_tangent, control_history = (
            simulate_gaussian_orbital_chain_second_order(
                **control_common, noise=control_noise, store_history=True
            )
        )
        exact_central = simulate_exact_small_chain(
            **control_common, noise=control_noise, store_covariance_history=False
        )
        exact_control = {
            "seconds_before_fd": perf_counter() - start,
            "base_z_rms_error": _rms(
                control_history.z - exact_central.z_history
            ),
            "base_final_covariance_rms_error": _rms(
                control_tangent.final_covariance
                - exact_central.final_covariance
            ),
            "growth": _growth_summary(
                control_tangent, control_history, args.dt, n_burnin
            ),
            "finite_difference": _finite_difference_sweep(
                runner=simulate_exact_small_chain,
                common=control_common,
                noise=control_noise,
                tangent=control_tangent,
                history=control_history,
                central_q=exact_central.q,
                central_covariance=exact_central.final_covariance,
                h_values=args.h_values,
                n_burnin=n_burnin,
            ),
        }
        print(
            "exact L=2 control: base z rms="
            f"{exact_control['base_z_rms_error']:.3e}; max(V2,F2)="
            f"({np.max(control_history.growth.second_orbital_norm):.3e}, "
            f"{np.max(control_history.growth.second_covariance_norm):.3e})"
        )

    primary_csv = args.output_dir / "primary_tangent_timeseries.csv"
    _write_timeseries(primary_csv, history, args.dt, args.record_stride, "primary")
    control_csv = None
    if control_history is not None:
        control_csv = args.output_dir / "exact_control_tangent_timeseries.csv"
        _write_timeseries(
            control_csv, control_history, args.dt, args.record_stride, "exact_control"
        )

    payload = {
        "schema_version": 1,
        "configuration": {
            "L": args.L,
            "g": args.g,
            "gamma": gamma,
            "J": args.J,
            "dt": args.dt,
            "burn_in_time": args.burn_in_time,
            "sample_time": args.sample_time,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_trajectories": args.n_trajectories,
            "boundary": args.boundary,
            "seed": args.seed,
            "h_values": args.h_values,
            "fd_trajectories": args.fd_trajectories,
            "exact_control_L": None if args.skip_exact_control else 2,
            "exact_control_trajectories": args.exact_control_trajectories,
            "record_stride": args.record_stride,
            "finite_difference_role": "common_noise_diagnostic_only",
            "repair_policy": "none",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git": _git_metadata(),
        },
        "primary": primary_summary,
        "primary_finite_difference": finite_difference,
        "exact_control": exact_control,
    }
    json_path = args.output_dir / "tangent_growth_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    hashes = {primary_csv.name: _sha256(primary_csv)}
    if control_csv is not None:
        hashes[control_csv.name] = _sha256(control_csv)
    hashes_path = args.output_dir / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {primary_csv}")
    if control_csv is not None:
        print(f"wrote {control_csv}")
    print(f"wrote {hashes_path}")


if __name__ == "__main__":
    main()
