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
from quantum_measurement.parallel.run_profiles import (
    LEGACY_PROFILE,
    TANGENT_GUARDED_PROFILE,
    TangentSchedule,
    resolve_tangent_schedule,
)

SCHEMA_VERSION = 1
CSV_FIELDS = (
    "gamma", "requested_dt", "dt", "h", "profile_name", "profile_version", "schedule_fingerprint",
    "decision_state", "decision_reason", "parent_fingerprint", "batch_index", "batch_seed", "J", "n_trajectories", "n_burnin", "n_samples",
    "dq_mean", "dq_sem", "d2q_mean", "d2q_sem", "fd_dq_mean", "fd_dq_sem",
    "fd_d2q_mean", "fd_d2q_sem", "dq_rms_error", "d2q_rms_error",
    "dQ_mean", "d2Q_mean", "fd_dQ_mean", "fd_d2Q_mean",
    "dQ_rms_error", "d2Q_rms_error", "tangent_seconds",
    "finite_difference_seconds", "runtime_ratio", "max_first_tangent_norm",
    "max_second_tangent_norm", "max_norm_error", "max_first_constraint_error",
    "max_second_constraint_error", "noise_hash", "seed",
)
AGGREGATE_FIELDS = (
    "gamma", "requested_dt", "dt", "h", "profile_name", "profile_version", "schedule_fingerprint",
    "decision_state", "n_batches", "dq_mean", "dq_sem", "d2q_mean", "d2q_sem",
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
    parser.add_argument(
        "--profile",
        choices=(LEGACY_PROFILE, TANGENT_GUARDED_PROFILE),
        default=LEGACY_PROFILE,
    )
    parser.add_argument("--profile-max-second-tangent-norm", type=float, default=1.0e3)
    parser.add_argument("--profile-max-second-tangent-sem", type=float, default=10.0)
    parser.add_argument("--profile-dt-min", type=float, default=1.0e-6)
    parser.add_argument("--profile-horizon-reduction", type=float, default=0.5)
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


def _load_records(path: Path) -> dict[str, dict[str, object]]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["key"])] = record["row"]
    return records


def _load_decisions(path: Path) -> dict[tuple[float, float, int], dict[str, object]]:
    decisions = {}
    if not path.exists():
        return decisions
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        decisions[
            (float(record["gamma"]), float(record["requested_dt"]), int(record["batch_index"]))
        ] = record
    return decisions


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_aggregates(records_path: Path, aggregate_path: Path) -> None:
    grouped: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)["row"]
            key = (
                float(record["gamma"]),
                float(record["requested_dt"]),
                float(record["dt"]),
                float(record["h"]),
                record["profile_name"],
                int(record["profile_version"]),
                record["schedule_fingerprint"],
                record["decision_state"],
            )
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
    for (gamma, requested_dt, dt, h, profile_name, profile_version, schedule_fingerprint, decision_state), records in sorted(grouped.items()):
        row = {
            "gamma": gamma,
            "requested_dt": requested_dt,
            "dt": dt,
            "h": h,
            "profile_name": profile_name,
            "profile_version": profile_version,
            "schedule_fingerprint": schedule_fingerprint,
            "decision_state": decision_state,
            "n_batches": len(records),
        }
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
    if (
        args.profile_max_second_tangent_norm <= 0.0
        or args.profile_max_second_tangent_sem <= 0.0
        or args.profile_dt_min <= 0.0
        or not 0.0 < args.profile_horizon_reduction <= 1.0
    ):
        raise ValueError("invalid adaptive profile guardrails")

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
        "profile": args.profile,
        "profile_max_second_tangent_norm": args.profile_max_second_tangent_norm,
        "profile_max_second_tangent_sem": args.profile_max_second_tangent_sem,
        "profile_dt_min": args.profile_dt_min,
        "profile_horizon_reduction": args.profile_horizon_reduction,
        "finite_difference_role": "short_common_noise_validation_only",
    }
    fingerprint = _fingerprint(configuration)
    configuration_path = args.output_dir / "configuration.json"
    manifest_path = args.output_dir / "manifest.json"
    rows_path = args.output_dir / "step_size_summary.csv"
    aggregate_path = args.output_dir / "aggregate_summary.csv"
    records_path = args.output_dir / "records.jsonl"
    decisions_path = args.output_dir / "profile_decisions.jsonl"

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
    existing_records = _load_records(records_path)
    decisions = _load_decisions(decisions_path)
    total_blocks = len(args.gamma_values) * len(args.dt_values) * len(args.h_values) * args.n_batches

    from quantum_measurement.aot_single_qubit import rademacher_noise

    block_number = 0
    for gamma_index, gamma in enumerate(args.gamma_values):
        for dt_index, requested_dt in enumerate(args.dt_values):
            for batch_index in range(args.n_batches):
                previous_decision = decisions.get((gamma, requested_dt, batch_index - 1))
                if previous_decision is not None:
                    schedule = TangentSchedule.from_dict(previous_decision["next_schedule"])
                elif batch_index == 0:
                    schedule = resolve_tangent_schedule(
                        gamma=gamma,
                        J=args.J,
                        dt=requested_dt,
                        n_burnin=args.n_burnin,
                        n_samples=args.n_samples,
                        profile_name=args.profile,
                    )
                elif args.profile == LEGACY_PROFILE:
                    schedule = resolve_tangent_schedule(
                        gamma=gamma,
                        J=args.J,
                        dt=requested_dt,
                        n_burnin=args.n_burnin,
                        n_samples=args.n_samples,
                        profile_name=args.profile,
                    )
                else:
                    raise RuntimeError(
                        f"missing profile decision for gamma={gamma:g}, requested_dt={requested_dt:g}, "
                        f"previous batch={batch_index - 1}"
                    )

                noise_seed = args.seed + gamma_index * 100003 + dt_index * 1009 + batch_index
                noise = rademacher_noise(
                    args.n_trajectories,
                    schedule.n_steps,
                    noise_seed,
                )
                batch_rows = []
                for h in args.h_values:
                    key = f"gamma={gamma:g}|dt={requested_dt:g}|h={h:g}|batch={batch_index}"
                    if args.profile != LEGACY_PROFILE:
                        key += f"|schedule={schedule.fingerprint}"
                    block_number += 1
                    if key in completed:
                        print(f"SKIP {block_number}/{total_blocks} {key}", flush=True)
                        if key not in existing_records:
                            raise RuntimeError(f"manifest references missing record for {key}")
                        batch_rows.append(existing_records[key])
                        continue

                    started = perf_counter()
                    validation = validate_aot_second_order_finite_difference(
                        gamma=gamma,
                        J=args.J,
                        dt=schedule.dt,
                        n_burnin=schedule.n_burnin,
                        n_samples=schedule.n_samples,
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
                        "gamma": gamma, "requested_dt": requested_dt, "dt": schedule.dt, "h": h,
                        "profile_name": schedule.profile_name,
                        "profile_version": schedule.profile_version,
                        "schedule_fingerprint": schedule.fingerprint,
                        "decision_state": schedule.decision_state,
                        "decision_reason": schedule.decision_reason,
                        "parent_fingerprint": schedule.parent_fingerprint,
                        "batch_index": batch_index,
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
                    _append_jsonl(records_path, record)
                    _append_csv(rows_path, row)
                    _write_aggregates(records_path, aggregate_path)
                    existing_records[key] = row
                    batch_rows.append(row)
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

                if args.profile != LEGACY_PROFILE:
                    diagnostics = {
                        "max_second_tangent_norm": max(
                            float(row["max_second_tangent_norm"]) for row in batch_rows
                        ),
                        "max_first_tangent_norm": max(
                            float(row["max_first_tangent_norm"]) for row in batch_rows
                        ),
                        "d2q_sem": max(float(row["d2q_sem"]) for row in batch_rows),
                        "max_second_constraint_error": max(
                            float(row["max_second_constraint_error"]) for row in batch_rows
                        ),
                    }
                    next_schedule = resolve_tangent_schedule(
                        gamma=gamma,
                        J=args.J,
                        dt=schedule.dt,
                        n_burnin=schedule.n_burnin,
                        n_samples=schedule.n_samples,
                        profile_name=args.profile,
                        diagnostics=diagnostics,
                        max_second_tangent_norm=args.profile_max_second_tangent_norm,
                        max_second_tangent_sem=args.profile_max_second_tangent_sem,
                        dt_min=args.profile_dt_min,
                        horizon_reduction=args.profile_horizon_reduction,
                    )
                    decision_key = (gamma, requested_dt, batch_index)
                    decision = {
                        "gamma": gamma,
                        "requested_dt": requested_dt,
                        "batch_index": batch_index,
                        "configuration_fingerprint": fingerprint,
                        "schedule": schedule.to_dict(),
                        "diagnostics": diagnostics,
                        "next_schedule": next_schedule.to_dict(),
                        "transition": next_schedule.decision_state,
                        "reason": next_schedule.decision_reason,
                    }
                    if decision_key not in decisions:
                        _append_jsonl(decisions_path, decision)
                        decisions[decision_key] = decision
                    manifest["last_profile_decision"] = decision
                    _atomic_write_json(manifest_path, manifest)

    manifest["status"] = "complete" if len(completed) == total_blocks else "partial"
    _atomic_write_json(manifest_path, manifest)
    print(f"STUDY_{manifest['status'].upper()} completed={len(completed)}/{total_blocks}", flush=True)


if __name__ == "__main__":
    main()
