#!/usr/bin/env python3
"""Validate the direct second-order single-qubit AoT tangent estimator.

The command uses common primitive noise only for short derivative checks.
Finite differences produced here are validation evidence, not production
derivative estimates.
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

import matplotlib.pyplot as plt
import numpy as np
import scipy

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from quantum_measurement.aot_single_qubit import (
    dressel_no_drive_reference,
    mean_and_sem,
    rademacher_noise,
    second_order_summary,
    validate_aot_second_order_finite_difference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gamma-values", nargs="+", type=float, default=[0.3, 1.0, 4.0]
    )
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        default=[2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3],
    )
    parser.add_argument("--dt-values", nargs="+", type=float, default=[0.005])
    parser.add_argument("--J", type=float, default=0.0)
    parser.add_argument("--n-trajectories", type=int, default=512)
    parser.add_argument("--n-burnin", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument(
        "--burn-in-time",
        type=float,
        default=None,
        help="Override n-burnin at each dt to keep physical burn-in time fixed.",
    )
    parser.add_argument(
        "--sample-time",
        type=float,
        default=None,
        help="Override n-samples at each dt to keep physical sample time fixed.",
    )
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_second_order_single_qubit_validation"),
    )
    return parser.parse_args()


def _count_for_time(duration: float, dt: float, name: str) -> int:
    count = int(round(duration / dt))
    if count < 0 or not np.isclose(count * dt, duration, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a nonnegative integer multiple of dt")
    return count


def _protocol_counts(args: argparse.Namespace, dt: float) -> tuple[int, int]:
    n_burnin = args.n_burnin
    n_samples = args.n_samples
    if args.burn_in_time is not None:
        n_burnin = _count_for_time(args.burn_in_time, dt, "burn-in-time")
    if args.sample_time is not None:
        n_samples = _count_for_time(args.sample_time, dt, "sample-time")
    if n_samples <= 0:
        raise ValueError("the effective n-samples must be positive")
    return n_burnin, n_samples


def _running_means(values: np.ndarray) -> list[dict[str, float]]:
    sizes = []
    size = 16
    while size < values.size:
        sizes.append(size)
        size *= 2
    sizes.append(values.size)
    records = []
    for size in sorted(set(sizes)):
        mean, sem = mean_and_sem(values[:size])
        records.append({"n": int(size), "mean": mean, "sem": sem})
    return records


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    mean, sem = mean_and_sem(values)
    return {
        "mean": mean,
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "sem": sem,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0.01, 0.1, 0.5, 0.9, 0.99),
                np.quantile(values, (0.01, 0.1, 0.5, 0.9, 0.99)),
            )
        },
        "running_means": _running_means(values),
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _sample_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


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


def _validate_arguments(args: argparse.Namespace) -> None:
    if any(value <= 0.0 for value in args.gamma_values):
        raise ValueError("all gamma-values must be positive")
    if any(value <= 0.0 for value in args.h_values):
        raise ValueError("all h-values must be positive")
    if any(value <= 0.0 for value in args.dt_values):
        raise ValueError("all dt-values must be positive")
    if args.J < 0.0:
        raise ValueError("J must be nonnegative")
    if args.n_trajectories <= 0:
        raise ValueError("n-trajectories must be positive")
    if args.n_burnin < 0 or args.n_samples <= 0:
        raise ValueError("n-burnin must be nonnegative and n-samples positive")
    if args.burn_in_time is not None and args.burn_in_time < 0.0:
        raise ValueError("burn-in-time must be nonnegative")
    if args.sample_time is not None and args.sample_time <= 0.0:
        raise ValueError("sample-time must be positive")


def main() -> None:
    args = parse_args()
    _validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_records = []
    flat_rows = []
    plot_samples = []
    run_index = 0
    for gamma in args.gamma_values:
        for dt in args.dt_values:
            n_burnin, n_samples = _protocol_counts(args, dt)
            n_steps = n_burnin + n_samples
            noise_seed = args.seed + run_index
            noise = rademacher_noise(args.n_trajectories, n_steps, noise_seed)
            run_index += 1
            validations = []
            elapsed = []
            for h in args.h_values:
                start = perf_counter()
                validation = validate_aot_second_order_finite_difference(
                    gamma=gamma,
                    J=args.J,
                    dt=dt,
                    n_burnin=n_burnin,
                    n_samples=n_samples,
                    h=h,
                    noise=noise,
                )
                elapsed.append(perf_counter() - start)
                validations.append(validation)

            central = validations[0].tangent
            summary = second_order_summary(central)
            distributions = {
                field_name: _distribution(getattr(central, field_name))
                for field_name in (
                    "q",
                    "dq_dtheta",
                    "d2q_dtheta2",
                    "Q",
                    "dQ_dtheta",
                    "d2Q_dtheta2",
                    "max_first_tangent_norm",
                    "max_second_tangent_norm",
                )
            }
            analytical = None
            if args.J == 0.0 and n_burnin == 0:
                analytical_reference = dressel_no_drive_reference(
                    gamma * central.averaging_time,
                    quadrature_order=args.quadrature_order,
                )
                analytical = {
                    "s": analytical_reference.s,
                    "mean_Q": analytical_reference.mean_Q,
                    "dQ_dtheta": analytical_reference.chi_Q,
                    "d2Q_dtheta2": analytical_reference.d2Q_dlog_s2,
                    "Q_z_score": (
                        (summary["Q_mean"] - analytical_reference.mean_Q)
                        / summary["Q_sem"]
                        if summary["Q_sem"] > 0.0
                        else None
                    ),
                    "dQ_z_score": (
                        (summary["dQ_dtheta_mean"] - analytical_reference.chi_Q)
                        / summary["dQ_dtheta_sem"]
                        if summary["dQ_dtheta_sem"] > 0.0
                        else None
                    ),
                    "d2Q_z_score": (
                        (
                            summary["d2Q_dtheta2_mean"]
                            - analytical_reference.d2Q_dlog_s2
                        )
                        / summary["d2Q_dtheta2_sem"]
                        if summary["d2Q_dtheta2_sem"] > 0.0
                        else None
                    ),
                }

            step_records = []
            for validation, seconds in zip(validations, elapsed):
                tangent_dq_std = _sample_std(validation.tangent.dq_dtheta)
                tangent_d2q_std = _sample_std(validation.tangent.d2q_dtheta2)
                fd_dq_std = _sample_std(validation.finite_difference_dq_dtheta)
                fd_d2q_std = _sample_std(validation.finite_difference_d2q_dtheta2)
                row = {
                    "gamma": gamma,
                    "J": args.J,
                    "g": gamma / (4.0 * args.J) if args.J > 0.0 else None,
                    "dt": dt,
                    "n_burnin": n_burnin,
                    "n_samples": n_samples,
                    "n_trajectories": args.n_trajectories,
                    "h": validation.h,
                    "u_rms_error": _rms(
                        validation.history.u - validation.finite_difference_u
                    ),
                    "v_rms_error": _rms(
                        validation.history.v - validation.finite_difference_v
                    ),
                    "dq_rms_error": _rms(
                        validation.tangent.dq_dtheta
                        - validation.finite_difference_dq_dtheta
                    ),
                    "d2q_rms_error": _rms(
                        validation.tangent.d2q_dtheta2
                        - validation.finite_difference_d2q_dtheta2
                    ),
                    "dQ_rms_error": _rms(
                        validation.tangent.dQ_dtheta
                        - validation.finite_difference_dQ_dtheta
                    ),
                    "d2Q_rms_error": _rms(
                        validation.tangent.d2Q_dtheta2
                        - validation.finite_difference_d2Q_dtheta2
                    ),
                    "validation_seconds": seconds,
                    "tangent_seconds": validation.tangent_seconds,
                    "finite_difference_seconds": validation.finite_difference_seconds,
                    "fd_to_tangent_runtime_ratio": (
                        validation.finite_difference_seconds
                        / validation.tangent_seconds
                        if validation.tangent_seconds > 0.0
                        else None
                    ),
                    "tangent_dq_std": tangent_dq_std,
                    "finite_difference_dq_std": fd_dq_std,
                    "tangent_to_fd_dq_variance_ratio": (
                        (tangent_dq_std / fd_dq_std) ** 2
                        if fd_dq_std > 0.0
                        else None
                    ),
                    "tangent_d2q_std": tangent_d2q_std,
                    "finite_difference_d2q_std": fd_d2q_std,
                    "tangent_to_fd_d2q_variance_ratio": (
                        (tangent_d2q_std / fd_d2q_std) ** 2
                        if fd_d2q_std > 0.0
                        else None
                    ),
                }
                step_records.append(row)
                flat_rows.append(row)

            record = {
                "gamma": gamma,
                "J": args.J,
                "g": gamma / (4.0 * args.J) if args.J > 0.0 else None,
                "log_gamma": float(np.log(gamma)),
                "log_g": central.log_g if args.J > 0.0 else None,
                "dt": dt,
                "n_burnin": n_burnin,
                "n_samples": n_samples,
                "averaging_time": central.averaging_time,
                "n_trajectories": central.n_trajectories,
                "noise_seed": noise_seed,
                "noise_hash": central.noise_hash,
                "derivative_parameter": central.derivative_parameter,
                "summary": summary,
                "distributions": distributions,
                "analytical_no_drive": analytical,
                "step_sizes": step_records,
                "runtime": {
                    "direct_tangent_seconds": validations[0].tangent_seconds,
                    "paired_physical_runs_seconds_by_h": [
                        validation.finite_difference_seconds
                        for validation in validations
                    ],
                },
            }
            run_records.append(record)
            plot_samples.append(
                (
                    f"g={gamma:g}\ndt={dt:g}",
                    np.array(central.dq_dtheta),
                    np.array(central.d2q_dtheta2),
                )
            )
            print(
                f"gamma={gamma:g}, dt={dt:g}: "
                f"d2Q={summary['d2Q_dtheta2_mean']:+.6g} "
                f"+/- {summary['d2Q_dtheta2_sem']:.3g}; "
                f"max constraints=({summary['max_first_constraint_error']:.2e}, "
                f"{summary['max_second_constraint_error']:.2e})"
            )

    configuration = {
        "gamma_values": args.gamma_values,
        "h_values": args.h_values,
        "dt_values": args.dt_values,
        "J": args.J,
        "n_trajectories": args.n_trajectories,
        "n_burnin": args.n_burnin,
        "n_samples": args.n_samples,
        "burn_in_time": args.burn_in_time,
        "sample_time": args.sample_time,
        "seed": args.seed,
        "quadrature_order": args.quadrature_order,
        "finite_difference_role": "short_common_noise_validation_only",
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "git": _git_metadata(),
    }
    json_path = args.output_dir / "validation_results.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configuration": configuration,
                "environment": environment,
                "results": run_records,
            },
            indent=2,
        )
        + "\n"
    )

    csv_path = args.output_dir / "step_size_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    convergence_figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for record in run_records:
        label = f"gamma={record['gamma']:g}, dt={record['dt']:g}"
        h_values = np.array([row["h"] for row in record["step_sizes"]])
        axes[0].loglog(
            h_values,
            [row["dq_rms_error"] for row in record["step_sizes"]],
            "o-",
            label=label,
        )
        axes[1].loglog(
            h_values,
            [row["d2q_rms_error"] for row in record["step_sizes"]],
            "o-",
            label=label,
        )
    axes[0].set_title("First-derivative verification")
    axes[1].set_title("Second-derivative verification")
    for axis, ylabel in zip(axes, ("RMS dq error", "RMS d2q error")):
        axis.set_xlabel("log-parameter step h")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    convergence_figure.tight_layout()
    convergence_figure.savefig(
        args.output_dir / "derivative_convergence.png", dpi=180
    )
    plt.close(convergence_figure)

    distribution_figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [sample[0] for sample in plot_samples]
    axes[0].boxplot([sample[1] for sample in plot_samples], labels=labels)
    axes[1].boxplot([sample[2] for sample in plot_samples], labels=labels)
    axes[0].set_title("Per-trajectory first-tangent distribution")
    axes[1].set_title("Per-trajectory second-tangent distribution")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("dq/dtheta")
    axes[1].set_ylabel("d2q/dtheta2")
    distribution_figure.tight_layout()
    distribution_figure.savefig(
        args.output_dir / "sampling_uncertainty.png", dpi=180
    )
    plt.close(distribution_figure)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
