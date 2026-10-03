#!/usr/bin/env python3
"""Validate orbital-tangent convergence under coupled timestep refinement.

All timesteps share one finest Gaussian Brownian path. Finite differences are
optional short validation references and are never production estimators.
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantum_measurement.jw_expansion.gaussian_orbital import (
    coarsen_standard_normal_noise,
    simulate_gaussian_orbital_chain,
    simulate_gaussian_orbital_chain_second_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L-values", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--g-values", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--finest-dt", type=float, default=0.00125)
    parser.add_argument("--coarsening-factors", nargs="+", type=int, default=[8, 4, 2, 1])
    parser.add_argument("--burn-in-time", type=float, default=0.1)
    parser.add_argument("--sample-time", type=float, default=0.4)
    parser.add_argument("--n-trajectories", type=int, default=16)
    parser.add_argument("--boundary", choices=("open", "periodic"), default="periodic")
    parser.add_argument("--finite-difference-h", type=float, default=0.005)
    parser.add_argument("--skip-finite-difference", action="store_true")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_gaussian_orbital_timestep_validation"),
    )
    return parser.parse_args()


def _step_count(duration: float, dt: float, name: str) -> int:
    count = int(round(duration / dt))
    if count < 0 or not np.isclose(count * dt, duration, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a nonnegative integer multiple of dt")
    return count


def _validate(args: argparse.Namespace) -> tuple[int, int, list[int]]:
    if not args.L_values or any(L < 1 for L in args.L_values):
        raise ValueError("all L-values must be positive")
    if not args.g_values or any(g <= 0.0 for g in args.g_values):
        raise ValueError("all g-values must be positive")
    if args.J <= 0.0 or args.finest_dt <= 0.0:
        raise ValueError("J and finest-dt must be positive")
    if args.n_trajectories < 2 or args.finite_difference_h <= 0.0:
        raise ValueError("need at least two trajectories and positive h")
    factors = sorted(set(args.coarsening_factors), reverse=True)
    if not factors or factors[-1] != 1 or any(factor < 1 for factor in factors):
        raise ValueError("coarsening-factors must be positive and include 1")
    for coarse, fine in zip(factors[:-1], factors[1:]):
        if coarse % fine or coarse // fine < 2:
            raise ValueError("successive coarsening-factors must be nested")
    burn_in = _step_count(args.burn_in_time, args.finest_dt, "burn-in-time")
    samples = _step_count(args.sample_time, args.finest_dt, "sample-time")
    if samples < 1 or any(burn_in % f or samples % f for f in factors):
        raise ValueError("all factors must divide burn-in and sample counts")
    return burn_in, samples, factors


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return {
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "sem": sem,
        "confidence_95_half_width": 1.96 * sem,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            str(q): float(value)
            for q, value in zip((0.05, 0.5, 0.95), np.quantile(values, (0.05, 0.5, 0.95)))
        },
    }


def _paired_shift(coarse: np.ndarray, fine: np.ndarray) -> dict[str, float]:
    difference = np.asarray(coarse, dtype=float) - np.asarray(fine, dtype=float)
    sem = float(np.std(difference, ddof=1) / np.sqrt(difference.size))
    return {
        "mean": float(np.mean(difference)),
        "sem": sem,
        "confidence_95_half_width": 1.96 * sem,
        "rms": _rms(difference),
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


def _finite_difference_errors(
    tangent,
    history,
    noise: np.ndarray,
    protocol: dict[str, object],
    h: float,
) -> dict[str, float]:
    gamma = protocol["gamma"]
    physical = {key: value for key, value in protocol.items() if key != "gamma"}
    plus = simulate_gaussian_orbital_chain(
        **physical, gamma=gamma * np.exp(h), noise=noise
    )
    minus = simulate_gaussian_orbital_chain(
        **physical, gamma=gamma * np.exp(-h), noise=noise
    )
    finite_u = (plus.z_history - minus.z_history) / (2.0 * h)
    finite_v = (plus.z_history - 2.0 * history.z + minus.z_history) / (h * h)
    finite_dq = (plus.q - minus.q) / (2.0 * h)
    finite_d2q = (plus.q - 2.0 * tangent.q + minus.q) / (h * h)
    finite_F = (plus.final_covariance - minus.final_covariance) / (2.0 * h)
    finite_F2 = (
        plus.final_covariance - 2.0 * tangent.final_covariance + minus.final_covariance
    ) / (h * h)
    return {
        "u_rms": _rms(history.u - finite_u),
        "v_rms": _rms(history.v - finite_v),
        "dq_rms": _rms(tangent.dq_dtheta - finite_dq),
        "d2q_rms": _rms(tangent.d2q_dtheta2 - finite_d2q),
        "F_rms": _rms(tangent.final_first_covariance - finite_F),
        "F2_rms": _rms(tangent.final_second_covariance - finite_F2),
    }


def _pair_record(coarse_factor, fine_factor, finest_dt, by_factor):
    coarse_tangent, coarse_history = by_factor[coarse_factor]
    fine_tangent, fine_history = by_factor[fine_factor]
    stride = coarse_factor // fine_factor
    return {
        "coarse_factor": coarse_factor,
        "fine_factor": fine_factor,
        "coarse_dt": coarse_factor * finest_dt,
        "fine_dt": fine_factor * finest_dt,
        "z_history_rms": _rms(coarse_history.z - fine_history.z[:, ::stride, :]),
        "u_history_rms": _rms(coarse_history.u - fine_history.u[:, ::stride, :]),
        "v_history_rms": _rms(coarse_history.v - fine_history.v[:, ::stride, :]),
        "q_shift": _paired_shift(coarse_tangent.q, fine_tangent.q),
        "dq_shift": _paired_shift(
            coarse_tangent.dq_dtheta, fine_tangent.dq_dtheta
        ),
        "d2q_shift": _paired_shift(
            coarse_tangent.d2q_dtheta2, fine_tangent.d2q_dtheta2
        ),
    }


def _observed_orders(pair_records) -> dict[str, list[float]]:
    output = {}
    for name in ("z_history_rms", "u_history_rms", "v_history_rms"):
        values = np.array([record[name] for record in pair_records])
        output[name] = [
            float(
                np.log(values[index] / values[index + 1])
                / np.log(
                    pair_records[index]["coarse_dt"]
                    / pair_records[index + 1]["coarse_dt"]
                )
            )
            for index in range(values.size - 1)
            if values[index] > 0.0 and values[index + 1] > 0.0
        ]
    return output


def main() -> None:
    args = parse_args()
    fine_burn_in, fine_samples, factors = _validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    rows = []

    for L_index, L in enumerate(args.L_values):
        for g_index, g_value in enumerate(args.g_values):
            gamma = 4.0 * args.J * g_value
            noise_seed = args.seed + 100 * L_index + g_index
            fine_noise = np.random.default_rng(noise_seed).standard_normal(
                (args.n_trajectories, fine_burn_in + fine_samples, L)
            )
            by_factor = {}
            timestep_records = []
            for factor in factors:
                dt = args.finest_dt * factor
                noise = coarsen_standard_normal_noise(fine_noise, factor)
                protocol = {
                    "L": L,
                    "gamma": gamma,
                    "J": args.J,
                    "dt": dt,
                    "n_burnin": fine_burn_in // factor,
                    "n_samples": fine_samples // factor,
                    "boundary": args.boundary,
                }
                start = perf_counter()
                tangent, history = simulate_gaussian_orbital_chain_second_order(
                    **protocol, noise=noise, store_history=True
                )
                tangent_seconds = perf_counter() - start
                finite_difference = None
                if not args.skip_finite_difference:
                    start = perf_counter()
                    finite_difference = _finite_difference_errors(
                        tangent,
                        history,
                        noise,
                        protocol,
                        args.finite_difference_h,
                    )
                    finite_difference["seconds"] = perf_counter() - start
                invariants = {
                    name: np.max(getattr(tangent, name), axis=0).tolist()
                    for name in (
                        "max_hermiticity_residual",
                        "max_particle_hole_residual",
                        "max_projector_residual",
                        "max_orbital_constraint_residual",
                    )
                }
                timestep = {
                    "factor": factor,
                    "dt": dt,
                    "n_burnin": protocol["n_burnin"],
                    "n_samples": protocol["n_samples"],
                    "noise_hash": tangent.noise_hash,
                    "q": _distribution(tangent.q),
                    "dq": _distribution(tangent.dq_dtheta),
                    "d2q": _distribution(tangent.d2q_dtheta2),
                    "max_first_tangent_norm": _distribution(
                        tangent.max_first_tangent_norm
                    ),
                    "max_second_tangent_norm": _distribution(
                        tangent.max_second_tangent_norm
                    ),
                    "minimum_qr_diagonal": float(
                        np.min(tangent.minimum_qr_diagonal)
                    ),
                    "invariant_maxima_by_order": invariants,
                    "repair_count": int(np.sum(tangent.repair_count)),
                    "finite_difference": finite_difference,
                    "tangent_seconds": tangent_seconds,
                }
                timestep_records.append(timestep)
                by_factor[factor] = (tangent, history)
                row = {
                    "L": L,
                    "g": g_value,
                    "gamma": gamma,
                    "factor": factor,
                    "dt": dt,
                    "q_mean": timestep["q"]["mean"],
                    "q_sem": timestep["q"]["sem"],
                    "dq_mean": timestep["dq"]["mean"],
                    "dq_sem": timestep["dq"]["sem"],
                    "d2q_mean": timestep["d2q"]["mean"],
                    "d2q_sem": timestep["d2q"]["sem"],
                    "max_first_tangent_norm": timestep[
                        "max_first_tangent_norm"
                    ]["maximum"],
                    "max_second_tangent_norm": timestep[
                        "max_second_tangent_norm"
                    ]["maximum"],
                    "repair_count": timestep["repair_count"],
                }
                if finite_difference is not None:
                    row.update(finite_difference)
                rows.append(row)

            pair_records = [
                _pair_record(coarse, fine, args.finest_dt, by_factor)
                for coarse, fine in zip(factors[:-1], factors[1:])
            ]
            records.append(
                {
                    "L": L,
                    "g": g_value,
                    "gamma": gamma,
                    "noise_seed": noise_seed,
                    "fine_noise_hash": by_factor[1][0].noise_hash,
                    "timesteps": timestep_records,
                    "successive_pairs": pair_records,
                    "observed_strong_orders": _observed_orders(pair_records),
                }
            )
            finest = timestep_records[-1]
            print(
                f"L={L}, g={g_value:g}: finest (q,dq,d2q)="
                f"({finest['q']['mean']:+.5g}, {finest['dq']['mean']:+.5g}, "
                f"{finest['d2q']['mean']:+.5g}); max tangent norms="
                f"({finest['max_first_tangent_norm']['maximum']:.3g}, "
                f"{finest['max_second_tangent_norm']['maximum']:.3g})"
            )

    payload = {
        "schema_version": 1,
        "configuration": {
            "L_values": args.L_values,
            "g_values": args.g_values,
            "J": args.J,
            "finest_dt": args.finest_dt,
            "coarsening_factors": factors,
            "burn_in_time": args.burn_in_time,
            "sample_time": args.sample_time,
            "n_trajectories": args.n_trajectories,
            "boundary": args.boundary,
            "finite_difference_h": (
                None if args.skip_finite_difference else args.finite_difference_h
            ),
            "seed": args.seed,
            "noise_coupling": "coarse xi=sum(fine xi)/sqrt(factor)",
            "expected_pathwise_order": "strong order one-half or better",
            "finite_difference_role": "short_common_noise_validation_only",
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
    json_path = args.output_dir / "timestep_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output_dir / "timestep_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
