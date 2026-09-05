#!/usr/bin/env python3
"""Validate direct Gaussian-orbital tangents against exact small systems.

Finite differences in this script use identical primitive Gaussian noise and
serve only as short verification references.  The reported production
derivatives always come from direct orbital and deterministic-QR tangent propagation.
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

import numpy as np
import scipy

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from quantum_measurement.jw_expansion.gaussian_orbital import (
    simulate_exact_small_chain,
    simulate_gaussian_orbital_chain_second_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L-values", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-burnin", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=35)
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--boundary", choices=("open", "periodic"), default="periodic")
    parser.add_argument(
        "--h-values", nargs="+", type=float, default=[0.04, 0.02, 0.01, 0.005]
    )
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_gaussian_orbital_second_order_validation"),
    )
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.L_values or any(L < 1 or L > 4 for L in args.L_values):
        raise ValueError("exact-reference L-values must lie between 1 and 4")
    if args.gamma <= 0.0 or not np.isfinite(args.gamma):
        raise ValueError("gamma must be positive and finite")
    if args.J < 0.0 or not np.isfinite(args.J):
        raise ValueError("J must be nonnegative and finite")
    if args.dt <= 0.0 or not np.isfinite(args.dt):
        raise ValueError("dt must be positive and finite")
    if args.n_burnin < 0 or args.n_samples <= 0:
        raise ValueError("n-burnin must be nonnegative and n-samples positive")
    if args.n_trajectories < 1:
        raise ValueError("n-trajectories must be positive")
    if not args.h_values or any(h <= 0.0 for h in args.h_values):
        raise ValueError("all h-values must be positive")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _maximum(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _error_record(direct: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(direct) - np.asarray(reference)
    return {"rms": _rms(difference), "maximum": _maximum(difference)}


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


def main() -> None:
    args = parse_args()
    _validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = args.n_burnin + args.n_samples
    records = []
    rows = []

    for index, L in enumerate(args.L_values):
        noise_seed = args.seed + index
        noise = np.random.default_rng(noise_seed).standard_normal(
            (args.n_trajectories, n_steps, L)
        )
        common = dict(
            L=L,
            gamma=args.gamma,
            J=args.J,
            dt=args.dt,
            n_burnin=args.n_burnin,
            n_samples=args.n_samples,
            boundary=args.boundary,
            noise=noise,
        )
        start = perf_counter()
        tangent, history = simulate_gaussian_orbital_chain_second_order(
            **common, store_history=True
        )
        tangent_seconds = perf_counter() - start
        start = perf_counter()
        exact_central = simulate_exact_small_chain(**common)
        central_seconds = perf_counter() - start
        central_errors = {
            "z": _error_record(history.z, exact_central.z_history),
            "q": _error_record(tangent.q, exact_central.q),
            "final_covariance": _error_record(
                tangent.final_covariance, exact_central.final_covariance
            ),
        }

        h_records = []
        for h in args.h_values:
            start = perf_counter()
            exact_plus = simulate_exact_small_chain(
                **dict(common, gamma=args.gamma * np.exp(h))
            )
            exact_minus = simulate_exact_small_chain(
                **dict(common, gamma=args.gamma * np.exp(-h))
            )
            finite_difference_seconds = perf_counter() - start
            finite_u = (
                exact_plus.z_history - exact_minus.z_history
            ) / (2.0 * h)
            finite_v = (
                exact_plus.z_history
                - 2.0 * exact_central.z_history
                + exact_minus.z_history
            ) / (h * h)
            finite_dq = (exact_plus.q - exact_minus.q) / (2.0 * h)
            finite_d2q = (
                exact_plus.q - 2.0 * exact_central.q + exact_minus.q
            ) / (h * h)
            finite_F = (
                exact_plus.final_covariance - exact_minus.final_covariance
            ) / (2.0 * h)
            finite_F2 = (
                exact_plus.final_covariance
                - 2.0 * exact_central.final_covariance
                + exact_minus.final_covariance
            ) / (h * h)
            averaging_time = args.n_samples * args.dt
            exact_plus_Q = (
                exact_plus.gamma * L * averaging_time * exact_plus.q
            )
            exact_central_Q = (
                args.gamma * L * averaging_time * exact_central.q
            )
            exact_minus_Q = (
                exact_minus.gamma * L * averaging_time * exact_minus.q
            )
            finite_dQ = (exact_plus_Q - exact_minus_Q) / (2.0 * h)
            finite_d2Q = (
                exact_plus_Q - 2.0 * exact_central_Q + exact_minus_Q
            ) / (h * h)
            errors = {
                "u": _error_record(history.u, finite_u),
                "v": _error_record(history.v, finite_v),
                "dq": _error_record(tangent.dq_dtheta, finite_dq),
                "d2q": _error_record(tangent.d2q_dtheta2, finite_d2q),
                "F": _error_record(tangent.final_first_covariance, finite_F),
                "F2": _error_record(tangent.final_second_covariance, finite_F2),
                "dQ": _error_record(tangent.dQ_dtheta, finite_dQ),
                "d2Q": _error_record(tangent.d2Q_dtheta2, finite_d2Q),
            }
            h_record = {
                "h": h,
                "errors": errors,
                "finite_difference_seconds": finite_difference_seconds,
            }
            h_records.append(h_record)
            row = {
                "L": L,
                "h": h,
                "base_z_rms_error": central_errors["z"]["rms"],
                "base_q_rms_error": central_errors["q"]["rms"],
                "u_rms_error": errors["u"]["rms"],
                "v_rms_error": errors["v"]["rms"],
                "dq_rms_error": errors["dq"]["rms"],
                "d2q_rms_error": errors["d2q"]["rms"],
                "F_rms_error": errors["F"]["rms"],
                "F2_rms_error": errors["F2"]["rms"],
                "dQ_rms_error": errors["dQ"]["rms"],
                "d2Q_rms_error": errors["d2Q"]["rms"],
            }
            rows.append(row)

        convergence = {}
        for name in ("u", "v", "dq", "d2q", "F", "F2", "dQ", "d2Q"):
            errors = np.array([record["errors"][name]["rms"] for record in h_records])
            convergence[name] = {
                "rms_errors": errors.tolist(),
                "successive_reduction_factors": (
                    errors[:-1] / np.maximum(errors[1:], np.finfo(float).tiny)
                ).tolist(),
            }
        invariant_maxima = {
            name: np.max(getattr(tangent, name), axis=0).tolist()
            for name in (
                "max_hermiticity_residual",
                "max_particle_hole_residual",
                "max_projector_residual",
                "max_orbital_constraint_residual",
            )
        }
        record = {
            "L": L,
            "noise_seed": noise_seed,
            "noise_hash": tangent.noise_hash,
            "central_errors": central_errors,
            "h_sweep": h_records,
            "convergence": convergence,
            "invariant_maxima_by_order": invariant_maxima,
            "maximum_first_tangent_norm": float(
                np.max(tangent.max_first_tangent_norm)
            ),
            "maximum_second_tangent_norm": float(
                np.max(tangent.max_second_tangent_norm)
            ),
            "minimum_qr_diagonal": float(np.min(tangent.minimum_qr_diagonal)),
            "repair_count": int(np.sum(tangent.repair_count)),
            "tangent_seconds": tangent_seconds,
            "exact_central_seconds": central_seconds,
        }
        records.append(record)
        print(
            f"L={L}: base z rms={central_errors['z']['rms']:.3e}; "
            f"finest u/v rms=({h_records[-1]['errors']['u']['rms']:.3e}, "
            f"{h_records[-1]['errors']['v']['rms']:.3e}); "
            f"max tangent norms=({record['maximum_first_tangent_norm']:.3g}, "
            f"{record['maximum_second_tangent_norm']:.3g})"
        )

    payload = {
        "schema_version": 1,
        "configuration": {
            "L_values": args.L_values,
            "gamma": args.gamma,
            "J": args.J,
            "g": args.gamma / (4.0 * args.J) if args.J > 0.0 else None,
            "dt": args.dt,
            "n_burnin": args.n_burnin,
            "n_samples": args.n_samples,
            "n_trajectories": args.n_trajectories,
            "boundary": args.boundary,
            "h_values": args.h_values,
            "seed": args.seed,
            "initial_state": "computational |1>^L (article z_x=-1)",
            "derivative_parameter": "log_gamma_at_fixed_J",
            "finite_difference_role": "short_common_noise_exact_validation_only",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git": _git_metadata(),
        },
        "results": records,
    }
    json_path = args.output_dir / "validation_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output_dir / "h_sweep_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
