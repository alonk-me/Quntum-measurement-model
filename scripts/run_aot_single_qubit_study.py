#!/usr/bin/env python3
"""Run a checkpointed single-qubit first/second derivative study.

Each completed ``(gamma, dt, h)`` block is appended to a JSONL and CSV
artifact and recorded in an atomic manifest. The driver is intentionally CPU
serial: it is a reproducible reference workflow, not a production scheduler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from time import perf_counter

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from quantum_measurement.aot_single_qubit import (
    mean_and_sem,
    validate_aot_second_order_finite_difference,
)

SCHEMA_VERSION = 1
CSV_FIELDS = (
    "gamma", "dt", "h", "batch_index", "batch_seed", "J", "n_trajectories", "n_burnin", "n_samples",
    "dq_mean", "dq_sem", "d2q_mean", "d2q_sem", "fd_dq_mean", "fd_dq_sem",
    "fd_d2q_mean", "fd_d2q_sem", "dq_rms_error", "d2q_rms_error",
    "dQ_mean", "d2Q_mean", "fd_dQ_mean", "fd_d2Q_mean",
    "dQ_rms_error", "d2Q_rms_error", "tangent_seconds",
    "finite_difference_seconds", "runtime_ratio", "max_first_tangent_norm",
    "max_second_tangent_norm", "max_norm_error", "max_first_constraint_error",
    "max_second_constraint_error", "noise_hash", "seed",
)
AGGREGATE_FIELDS = (
    "gamma", "dt", "h", "n_batches", "dq_mean", "dq_sem", "d2q_mean", "d2q_sem",
    "fd_dq_mean", "fd_dq_sem", "fd_d2q_mean", "fd_d2q_sem", "dq_rms_error",
    "d2q_rms_error", "dQ_mean", "d2Q_mean", "fd_dQ_mean", "fd_d2Q_mean",
    "dQ_rms_error", "d2Q_rms_error", "max_first_tangent_norm", "max_second_tangent_norm",
    "max_norm_error", "max_first_constraint_error", "max_second_constraint_error",
    "tangent_seconds", "finite_difference_seconds", "runtime_ratio",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma-values", nargs="+", type=float, required=True)
    parser.add_argument("--dt-values", nargs="+", type=float, default=[0.005])
    parser.add_argument("--h-values", nargs="+", type=float, default=[0.02, 0.01, 0.005])
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=1)
    parser.add_argument("--n-burnin", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-first-tangent-norm", type=float, default=1.0e6)
    parser.add_argument("--max-second-tangent-norm", type=float, default=1.0e8)
    return parser.parse_args()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fingerprint(configuration: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()


def _noise_hash(noise: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(noise).view(np.uint8))
    return digest.hexdigest()


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    return mean_and_sem(np.asarray(values, dtype=float))


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_csv(path: Path, row: dict[str, object]) -> None:
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _write_aggregates(records_path: Path, aggregate_path: Path) -> None:
    grouped: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)["row"]
            key = (float(record["gamma"]), float(record["dt"]), float(record["h"]))
            grouped.setdefault(key, []).append(record)

    rows = []
    mean_fields = (
        "dq_mean", "d2q_mean", "fd_dq_mean", "fd_d2q_mean", "dq_rms_error",
        "d2q_rms_error", "dQ_mean", "d2Q_mean", "fd_dQ_mean", "fd_d2Q_mean",
        "dQ_rms_error", "d2Q_rms_error", "max_first_tangent_norm",
        "max_second_tangent_norm", "max_norm_error", "max_first_constraint_error",
        "max_second_constraint_error", "tangent_seconds", "finite_difference_seconds",
        "runtime_ratio",
    )
    for (gamma, dt, h), records in sorted(grouped.items()):
        row = {"gamma": gamma, "dt": dt, "h": h, "n_batches": len(records)}
        for field in mean_fields:
            values = np.asarray([float(record[field]) for record in records])
            row[field] = float(np.mean(values))
        for field in ("dq_mean", "d2q_mean", "fd_dq_mean", "fd_d2q_mean"):
            values = np.asarray([float(record[field]) for record in records])
            row[field.replace("_mean", "_sem")] = float(
                np.std(values, ddof=1) / np.sqrt(values.size)
            ) if values.size > 1 else 0.0
        rows.append(row)

    temporary = aggregate_path.with_suffix(aggregate_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(aggregate_path)


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def main() -> None:
    args = _parse_args()
    if any(value <= 0 for value in args.gamma_values + args.dt_values + args.h_values):
        raise ValueError("gamma, dt, and h values must be positive")
    if args.J < 0 or args.n_trajectories <= 0 or args.n_batches <= 0 or args.n_burnin < 0 or args.n_samples <= 0:
        raise ValueError("invalid J, trajectory, burn-in, or sample configuration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "gamma_values": args.gamma_values,
        "dt_values": args.dt_values,
        "h_values": args.h_values,
        "J": args.J,
        "n_trajectories": args.n_trajectories,
        "n_batches": args.n_batches,
        "n_burnin": args.n_burnin,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "finite_difference_role": "short_common_noise_validation_only",
    }
    fingerprint = _fingerprint(configuration)
    configuration_path = args.output_dir / "configuration.json"
    manifest_path = args.output_dir / "manifest.json"
    rows_path = args.output_dir / "step_size_summary.csv"
    aggregate_path = args.output_dir / "aggregate_summary.csv"
    records_path = args.output_dir / "records.jsonl"

    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("output directory has an incompatible configuration fingerprint")
    else:
        _atomic_write_json(
            configuration_path,
            {"configuration": configuration, "fingerprint": fingerprint, "environment": _environment()},
        )

    manifest = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "completed": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError("manifest fingerprint does not match configuration")
    completed = set(manifest.get("completed", []))
    total_blocks = len(args.gamma_values) * len(args.dt_values) * len(args.h_values) * args.n_batches

    from quantum_measurement.aot_single_qubit import rademacher_noise

    block_number = 0
    for gamma_index, gamma in enumerate(args.gamma_values):
        for dt_index, dt in enumerate(args.dt_values):
            for batch_index in range(args.n_batches):
                noise_seed = args.seed + gamma_index * 100003 + dt_index * 1009 + batch_index
                noise = rademacher_noise(
                    args.n_trajectories,
                    args.n_burnin + args.n_samples,
                    noise_seed,
                )
                for h in args.h_values:
                    key = f"gamma={gamma:g}|dt={dt:g}|h={h:g}|batch={batch_index}"
                    block_number += 1
                    if key in completed:
                        print(f"SKIP {block_number}/{total_blocks} {key}", flush=True)
                        continue

                    started = perf_counter()
                    validation = validate_aot_second_order_finite_difference(
                        gamma=gamma,
                        J=args.J,
                        dt=dt,
                        n_burnin=args.n_burnin,
                        n_samples=args.n_samples,
                        h=h,
                        noise=noise,
                    )
                    elapsed = perf_counter() - started
                    tangent = validation.tangent
                    dq_mean, dq_sem = _mean_sem(tangent.dq_dtheta)
                    d2q_mean, d2q_sem = _mean_sem(tangent.d2q_dtheta2)
                    fd_dq_mean, fd_dq_sem = _mean_sem(validation.finite_difference_dq_dtheta)
                    fd_d2q_mean, fd_d2q_sem = _mean_sem(validation.finite_difference_d2q_dtheta2)
                    dq_error = tangent.dq_dtheta - validation.finite_difference_dq_dtheta
                    d2q_error = tangent.d2q_dtheta2 - validation.finite_difference_d2q_dtheta2
                    dQ_error = tangent.dQ_dtheta - validation.finite_difference_dQ_dtheta
                    d2Q_error = tangent.d2Q_dtheta2 - validation.finite_difference_d2Q_dtheta2
                    row = {
                        "gamma": gamma, "dt": dt, "h": h, "batch_index": batch_index,
                        "batch_seed": noise_seed, "J": args.J,
                        "n_trajectories": args.n_trajectories, "n_burnin": args.n_burnin,
                        "n_samples": args.n_samples, "dq_mean": dq_mean, "dq_sem": dq_sem,
                        "d2q_mean": d2q_mean, "d2q_sem": d2q_sem, "fd_dq_mean": fd_dq_mean,
                        "fd_dq_sem": fd_dq_sem, "fd_d2q_mean": fd_d2q_mean, "fd_d2q_sem": fd_d2q_sem,
                        "dq_rms_error": float(np.sqrt(np.mean(dq_error ** 2))),
                        "d2q_rms_error": float(np.sqrt(np.mean(d2q_error ** 2))),
                        "dQ_mean": float(np.mean(tangent.dQ_dtheta)),
                        "d2Q_mean": float(np.mean(tangent.d2Q_dtheta2)),
                        "fd_dQ_mean": float(np.mean(validation.finite_difference_dQ_dtheta)),
                        "fd_d2Q_mean": float(np.mean(validation.finite_difference_d2Q_dtheta2)),
                        "dQ_rms_error": float(np.sqrt(np.mean(dQ_error ** 2))),
                        "d2Q_rms_error": float(np.sqrt(np.mean(d2Q_error ** 2))),
                        "tangent_seconds": validation.tangent_seconds,
                        "finite_difference_seconds": validation.finite_difference_seconds,
                        "runtime_ratio": validation.finite_difference_seconds / validation.tangent_seconds,
                        "max_first_tangent_norm": float(np.max(tangent.max_first_tangent_norm)),
                        "max_second_tangent_norm": float(np.max(tangent.max_second_tangent_norm)),
                        "max_norm_error": float(np.max(tangent.max_norm_error)),
                        "max_first_constraint_error": float(np.max(tangent.max_first_constraint_error)),
                        "max_second_constraint_error": float(np.max(tangent.max_second_constraint_error)),
                        "noise_hash": tangent.noise_hash, "seed": noise_seed,
                    }
                    if row["max_first_tangent_norm"] > args.max_first_tangent_norm:
                        raise RuntimeError(f"first tangent guardrail exceeded at {key}")
                    if row["max_second_tangent_norm"] > args.max_second_tangent_norm:
                        raise RuntimeError(f"second tangent guardrail exceeded at {key}")
                    record = {"key": key, "row": row, "configuration_fingerprint": fingerprint}
                    with records_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    _append_csv(rows_path, row)
                    _write_aggregates(records_path, aggregate_path)
                    completed.add(key)
                    manifest["completed"] = sorted(completed)
                    manifest["last_completed"] = key
                    manifest["last_update_unix"] = time.time()
                    _atomic_write_json(manifest_path, manifest)
                    print(
                        f"COMPLETED {len(completed)}/{total_blocks} {key} "
                        f"dq={dq_mean:+.6g} d2q={d2q_mean:+.6g} "
                        f"dq_rms={row['dq_rms_error']:.3g} d2q_rms={row['d2q_rms_error']:.3g}",
                        flush=True,
                    )

    manifest["status"] = "complete" if len(completed) == total_blocks else "partial"
    _atomic_write_json(manifest_path, manifest)
    print(f"STUDY_{manifest['status'].upper()} completed={len(completed)}/{total_blocks}", flush=True)


if __name__ == "__main__":
    main()
