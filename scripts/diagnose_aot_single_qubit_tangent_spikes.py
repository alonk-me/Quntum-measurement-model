#!/usr/bin/env python3
"""Diagnose heavy-tailed single-qubit second tangents without trimming them.

The default protocol replays batch 0 of the 2026-10-03 ``gamma=4`` duration
study, whose direct ``d2q`` mean was dominated by one trajectory.  The
single-qubit map has no orbital gauge, so this control separates intrinsic
pathwise sensitivity from Gaussian-orbital coordinate effects.  Common-noise
finite differences are short diagnostic references only, swept towards small
``h`` to test whether they converge to the direct tangent.
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
import scipy.stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantum_measurement.aot_single_qubit import (
    rademacher_noise,
    validate_aot_second_order_finite_difference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--n-burnin", type=int, default=400)
    parser.add_argument("--n-samples", type=int, default=3200)
    parser.add_argument("--n-trajectories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        default=[0.02, 0.005, 0.001, 0.00025, 0.0001],
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--window-steps", type=int, default=200)
    parser.add_argument("--jump-window-steps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_tangent_growth_single_qubit"),
    )
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.gamma <= 0.0 or args.J < 0.0 or args.dt <= 0.0:
        raise ValueError("gamma and dt must be positive, and J nonnegative")
    if args.n_burnin < 0 or args.n_samples < 1 or args.n_trajectories < 2:
        raise ValueError("require n-burnin >= 0, n-samples >= 1, n-trajectories >= 2")
    if not args.h_values or any(h <= 0.0 for h in args.h_values):
        raise ValueError("all h-values must be positive")
    if list(args.h_values) != sorted(args.h_values, reverse=True):
        raise ValueError("h-values must be in decreasing order")
    if not 1 <= args.top_k <= args.n_trajectories:
        raise ValueError("top-k must lie between one and n-trajectories")
    if args.window_steps < 1 or args.jump_window_steps < 1:
        raise ValueError("window sizes must be positive")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    return {
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "sem": float(np.std(values, ddof=1) / np.sqrt(values.size)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "median": float(np.median(values)),
    }


def _agreement_horizon(
    direct: np.ndarray, reference: np.ndarray, dt: float, threshold: float = 0.1
) -> list[object]:
    """First time the relative pathwise disagreement exceeds ``threshold``."""

    scale = np.maximum(1.0, np.maximum(np.abs(direct), np.abs(reference)))
    normalized = np.abs(direct - reference) / scale
    output = []
    for trajectory in normalized:
        indices = np.flatnonzero(trajectory > threshold)
        output.append(None if not indices.size else float(indices[0] * dt))
    return output


def _windowed_maximum(values: np.ndarray, window: int) -> np.ndarray:
    n_windows = values.shape[1] // window
    trimmed = values[:, : n_windows * window]
    return trimmed.reshape(values.shape[0], n_windows, window).max(axis=2)


def _jump_alignment(z: np.ndarray, v: np.ndarray, window: int) -> dict[str, object]:
    """Rank-correlate windowed ``max|v|`` with fast ``z`` transitions."""

    step_jump = np.zeros_like(z)
    step_jump[:, 1:] = np.abs(np.diff(z, axis=1))
    jump = _windowed_maximum(step_jump, window)
    second = _windowed_maximum(np.abs(v), window)
    correlations = []
    for jump_values, second_values in zip(jump, second):
        if jump_values.size < 3 or np.ptp(jump_values) == 0.0 or np.ptp(second_values) == 0.0:
            correlations.append(None)
            continue
        correlations.append(
            float(scipy.stats.spearmanr(jump_values, second_values).correlation)
        )
    peak_windows = np.argmax(second, axis=1)
    return {
        "window_steps": window,
        "spearman_max_abs_v_vs_max_abs_dz": correlations,
        "jump_quantile_at_max_abs_v_window": [
            float(np.mean(trajectory <= trajectory[index]))
            for trajectory, index in zip(jump, peak_windows)
        ],
    }


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
    _validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = args.n_burnin + args.n_samples
    noise = rademacher_noise(args.n_trajectories, n_steps, args.seed)
    retained = slice(args.n_burnin, None)

    validations = []
    for h in args.h_values:
        start = perf_counter()
        validation = validate_aot_second_order_finite_difference(
            gamma=args.gamma,
            J=args.J,
            dt=args.dt,
            n_burnin=args.n_burnin,
            n_samples=args.n_samples,
            h=h,
            noise=noise,
        )
        validations.append((validation, perf_counter() - start))
        print(f"h={h:g}: {validations[-1][1]:.2f}s", flush=True)

    tangent = validations[0][0].tangent
    history = validations[0][0].history
    for validation, _ in validations[1:]:
        if not np.array_equal(validation.tangent.d2q_dtheta2, tangent.d2q_dtheta2):
            raise RuntimeError("direct tangent replay is not bitwise reproducible")

    z, u, v = history.z, history.u, history.v
    positive = np.mean(2.0 * np.square(u[:, retained]), axis=1)
    signed = np.mean(2.0 * z[:, retained] * v[:, retained], axis=1)
    peak_steps = np.argmax(np.abs(v), axis=1)
    rows = np.arange(args.n_trajectories)
    d2q = tangent.d2q_dtheta2
    largest = int(np.argmax(np.abs(d2q)))

    h_records = []
    for validation, seconds in validations:
        finite_d2q = validation.finite_difference_d2q_dtheta2
        h_records.append(
            {
                "h": validation.h,
                "seconds": seconds,
                "u_rms_error": _rms(u - validation.finite_difference_u),
                "v_rms_error": _rms(v - validation.finite_difference_v),
                "retained_v_rms_error": _rms(
                    v[:, retained] - validation.finite_difference_v[:, retained]
                ),
                "dq_abs_error": np.abs(
                    tangent.dq_dtheta - validation.finite_difference_dq_dtheta
                ).tolist(),
                "d2q_abs_error": np.abs(d2q - finite_d2q).tolist(),
                "finite_d2q": finite_d2q.tolist(),
                "u_agreement_horizon_10pct": _agreement_horizon(
                    u, validation.finite_difference_u, args.dt
                ),
                "v_agreement_horizon_10pct": _agreement_horizon(
                    v, validation.finite_difference_v, args.dt
                ),
            }
        )
    successive = [
        {
            "coarse_h": coarse[0].h,
            "fine_h": fine[0].h,
            "d2q_estimate_abs_difference": np.abs(
                coarse[0].finite_difference_d2q_dtheta2
                - fine[0].finite_difference_d2q_dtheta2
            ).tolist(),
            "v_estimate_rms_difference": _rms(
                coarse[0].finite_difference_v - fine[0].finite_difference_v
            ),
        }
        for coarse, fine in zip(validations[:-1], validations[1:])
    ]

    top = [int(index) for index in np.argsort(np.abs(d2q))[::-1][: args.top_k]]
    trace_path = args.output_dir / "top_trajectory_traces.csv"
    fields = ["trajectory", "step", "time", "retained", "xi", "z", "u", "v"] + [
        f"fd_v_h{validation.h:g}" for validation, _ in validations
    ]
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trajectory in top:
            peak = int(peak_steps[trajectory])
            first = max(0, peak - args.window_steps)
            last = min(n_steps, peak + args.window_steps + 1)
            for step in range(first, last):
                row = {
                    "trajectory": trajectory,
                    "step": step,
                    "time": step * args.dt,
                    "retained": int(step >= args.n_burnin),
                    "xi": int(noise[trajectory, step]),
                    "z": z[trajectory, step],
                    "u": u[trajectory, step],
                    "v": v[trajectory, step],
                }
                for validation, _ in validations:
                    row[f"fd_v_h{validation.h:g}"] = (
                        validation.finite_difference_v[trajectory, step]
                    )
                writer.writerow(row)

    payload = {
        "schema_version": 1,
        "configuration": {
            "gamma": args.gamma,
            "J": args.J,
            "dt": args.dt,
            "n_burnin": args.n_burnin,
            "n_samples": args.n_samples,
            "n_trajectories": args.n_trajectories,
            "seed": args.seed,
            "noise_kind": tangent.noise_kind,
            "noise_hash": tangent.noise_hash,
            "h_values": args.h_values,
            "top_k": args.top_k,
            "window_steps": args.window_steps,
            "jump_window_steps": args.jump_window_steps,
            "finite_difference_role": "common_noise_diagnostic_only",
            "derivative_parameter": tangent.derivative_parameter,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git": _git_metadata(),
        },
        "direct": {
            "q": _distribution(tangent.q),
            "dq": _distribution(tangent.dq_dtheta),
            "d2q": _distribution(d2q),
            "d2q_positive_2u2": _distribution(positive),
            "d2q_signed_2zv": _distribution(signed),
            "d2q_reconstruction_error": float(np.max(np.abs(d2q - positive - signed))),
            "largest_abs_d2q_trajectory": largest,
            "largest_abs_d2q_share_of_abs_sum": float(
                np.abs(d2q[largest]) / np.sum(np.abs(d2q))
            ),
            "max_first_tangent_norm": tangent.max_first_tangent_norm.tolist(),
            "max_second_tangent_norm": tangent.max_second_tangent_norm.tolist(),
            "max_second_constraint_error": tangent.max_second_constraint_error.tolist(),
            "peak_abs_v": np.abs(v[rows, peak_steps]).tolist(),
            "peak_abs_v_time": (peak_steps * args.dt).tolist(),
            "z_at_peak_abs_v": z[rows, peak_steps].tolist(),
            "u_at_peak_abs_v": u[rows, peak_steps].tolist(),
            "jump_alignment": _jump_alignment(z, v, args.jump_window_steps),
            "top_trajectories": top,
        },
        "finite_difference": {
            "h_records": h_records,
            "successive_estimate_differences": successive,
        },
    }
    json_path = args.output_dir / "tangent_spike_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    hashes_path = args.output_dir / "artifact_hashes.json"
    hashes_path.write_text(
        json.dumps({trace_path.name: _sha256(trace_path)}, indent=2) + "\n"
    )
    print(
        f"d2q mean={np.mean(d2q):.4g} sem={np.std(d2q, ddof=1) / np.sqrt(d2q.size):.4g}; "
        f"largest trajectory {largest} d2q={d2q[largest]:.4g}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
