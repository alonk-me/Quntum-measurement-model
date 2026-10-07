#!/usr/bin/env python3
"""Large-sample test of whether direct tangent derivatives average stably.

Per-trajectory ``q``, ``dq_dtheta``, and ``d2q_dtheta2`` are generated in
independent, seeded chunks with the accepted serial CPU estimators:
``simulate_gaussian_orbital_chain_second_order`` (``--system chain``) or
``simulate_aot_second_order_cpu`` (``--system single-qubit``).  Chunks are
reproducible and resumable; parallel workers only run independent chunks, so
results do not depend on the worker count.  No trajectory is trimmed.

``--reference`` adds a sampling-free stationary single-qubit reference from the
deterministic transfer operator.  It is a reference curve, not a trajectory
finite-difference estimator.
"""

from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import hashlib
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time
from time import perf_counter

import numpy as np
import scipy

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantum_measurement.aot_single_qubit import (
    rademacher_noise,
    simulate_aot_second_order_cpu,
    stationary_aot_density,
)
from quantum_measurement.jw_expansion.gaussian_orbital import (
    simulate_gaussian_orbital_chain_second_order,
)

SCHEMA_VERSION = 1
ARRAY_FIELDS = (
    "q",
    "dq_dtheta",
    "d2q_dtheta2",
    "max_first_tangent_norm",
    "max_second_tangent_norm",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("chain", "single-qubit"), required=True)
    parameters = parser.add_mutually_exclusive_group(required=True)
    parameters.add_argument("--g-values", nargs="+", type=float, help="g = gamma/(4J); requires J > 0")
    parameters.add_argument(
        "--gamma-values",
        nargs="+",
        type=float,
        help="measurement rates gamma directly; allows J = 0 (no-drive Dressel case)",
    )
    parser.add_argument(
        "--save-final-z",
        action="store_true",
        help="also store per-trajectory final z (needed for terminal ln R histograms)",
    )
    parser.add_argument("--n-trajectories", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--first-chunk-size",
        type=int,
        default=None,
        help="size of chunk 0 (default: 8 for chain, to replay the WBS 5.5 sentinel)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--burn-in-time", type=float, default=5.0)
    parser.add_argument("--sample-time", type=float, default=10.0)
    parser.add_argument("--boundary", choices=("open", "periodic"), default="periodic")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--reference-grids", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--reference-h", nargs="+", type=float, default=[0.1, 0.05])
    parser.add_argument("--reference-rtol", type=float, default=0.05)
    parser.add_argument("--reference-atol", type=float, default=1.0e-3)
    parser.add_argument("--reference-timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _step_count(duration: float, dt: float, name: str) -> int:
    count = int(round(duration / dt))
    if count < 0 or not np.isclose(count * dt, duration, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a nonnegative integer multiple of dt")
    return count


def _validate(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.first_chunk_size is None:
        args.first_chunk_size = 8 if args.system == "chain" else args.chunk_size
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.g_values is not None:
        if any(g <= 0.0 for g in args.g_values):
            raise ValueError("g-values must be positive")
        if args.J <= 0.0:
            raise ValueError("--g-values requires J > 0 (g is undefined at J=0)")
    else:
        if any(gamma <= 0.0 for gamma in args.gamma_values):
            raise ValueError("gamma-values must be positive")
        if args.J < 0.0:
            raise ValueError("J must be nonnegative")
    if args.n_trajectories < 2 or args.chunk_size < 1 or args.first_chunk_size < 1:
        raise ValueError("require n-trajectories >= 2 and positive chunk sizes")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.system == "chain" and args.L < 1:
        raise ValueError("L must be positive")
    if args.reference and (args.system != "single-qubit" or args.g_values is None):
        raise ValueError("--reference is available only for the driven single qubit with --g-values")
    n_burnin = _step_count(args.burn_in_time, args.dt, "burn-in-time")
    n_samples = _step_count(args.sample_time, args.dt, "sample-time")
    if n_samples < 1:
        raise ValueError("sample-time must include at least one step")
    return n_burnin, n_samples, n_burnin + n_samples


def _parameter_points(args: argparse.Namespace) -> list[tuple[object, float]]:
    """Return ``(g or None, gamma)`` for each requested parameter point."""

    if args.g_values is not None:
        return [(float(g), float(4.0 * args.J * g)) for g in args.g_values]
    return [(None, float(gamma)) for gamma in args.gamma_values]


def _chunk_specs(args: argparse.Namespace, n_burnin: int, n_samples: int) -> list[dict]:
    specs = []
    for g_index, (g, gamma) in enumerate(_parameter_points(args)):
        remaining = args.n_trajectories
        chunk = 0
        while remaining > 0:
            size = args.first_chunk_size if chunk == 0 else args.chunk_size
            size = min(size, remaining)
            seed = (
                args.seed + g_index
                if chunk == 0
                else args.seed + 1000 * (g_index + 1) + chunk
            )
            specs.append(
                {
                    "key": f"g{g_index}_c{chunk:05d}",
                    "system": args.system,
                    "g_index": g_index,
                    "g": g,
                    "gamma": gamma,
                    "J": float(args.J),
                    "save_final_z": bool(args.save_final_z),
                    "dt": float(args.dt),
                    "L": int(args.L),
                    "boundary": args.boundary,
                    "n_burnin": n_burnin,
                    "n_samples": n_samples,
                    "chunk": chunk,
                    "size": size,
                    "seed": seed,
                }
            )
            remaining -= size
            chunk += 1
    return specs


def run_chunk(spec: dict) -> dict:
    """Run one chunk with the serial reference estimator for its own noise."""

    steps = spec["n_burnin"] + spec["n_samples"]
    start = perf_counter()
    if spec["system"] == "chain":
        noise = np.random.default_rng(spec["seed"]).standard_normal(
            (spec["size"], steps, spec["L"])
        )
        result, _ = simulate_gaussian_orbital_chain_second_order(
            L=spec["L"],
            gamma=spec["gamma"],
            J=spec["J"],
            dt=spec["dt"],
            n_burnin=spec["n_burnin"],
            n_samples=spec["n_samples"],
            boundary=spec["boundary"],
            noise=noise,
        )
    else:
        noise = rademacher_noise(spec["size"], steps, spec["seed"])
        result = simulate_aot_second_order_cpu(
            gamma=spec["gamma"],
            J=spec["J"],
            dt=spec["dt"],
            n_burnin=spec["n_burnin"],
            n_samples=spec["n_samples"],
            noise=noise,
        )
    fields = ARRAY_FIELDS + (("final_z",) if spec["save_final_z"] else ())
    arrays = {name: np.asarray(getattr(result, name)) for name in fields}
    return {
        "arrays": arrays,
        "noise_hash": result.noise_hash,
        "noise_kind": result.noise_kind,
        "seconds": perf_counter() - start,
    }


def _reference_point(g: float, n_grid: int, J: float, dt: float) -> float:
    return stationary_aot_density(g, n_grid=n_grid, J=J, dt=dt)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_chunk(path: Path, spec: dict, output: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, seed=spec["seed"], **output["arrays"])
    temporary.replace(path)


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


def _configuration(args: argparse.Namespace, n_burnin: int, n_samples: int) -> dict:
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "system": args.system,
        "n_trajectories": args.n_trajectories,
        "chunk_size": args.chunk_size,
        "first_chunk_size": args.first_chunk_size,
        "seed": args.seed,
        "seed_rule": "chunk 0: seed+g_index; chunk c>0: seed+1000*(g_index+1)+c",
        "J": args.J,
        "dt": args.dt,
        "burn_in_time": args.burn_in_time,
        "sample_time": args.sample_time,
        "n_burnin": n_burnin,
        "n_samples": n_samples,
        "derivative_parameter": "log_gamma_at_fixed_J",
        "noise_kind": "standard_normal" if args.system == "chain" else "rademacher",
        "trimming": "none",
    }
    # Keys are added only when used, so existing run directories keep their
    # configuration fingerprint and remain resumable.
    if args.g_values is not None:
        configuration["g_values"] = [float(g) for g in args.g_values]
    else:
        configuration["gamma_values"] = [float(gamma) for gamma in args.gamma_values]
    if args.save_final_z:
        configuration["save_final_z"] = True
    if args.system == "chain":
        configuration.update({"L": args.L, "boundary": args.boundary})
    return configuration


def _run_chunks(args, specs, chunk_dir, manifest, manifest_path) -> None:
    completed = manifest["chunks"]
    pending = []
    for spec in specs:
        path = chunk_dir / f"{spec['key']}.npz"
        if spec["key"] in completed and path.exists():
            if _sha256(path) != completed[spec["key"]]["sha256"]:
                raise RuntimeError(f"checksum mismatch for completed chunk {spec['key']}")
            continue
        pending.append(spec)
    print(
        f"chunks: {len(specs) - len(pending)} complete, {len(pending)} pending",
        flush=True,
    )

    def record(spec: dict, output: dict) -> None:
        path = chunk_dir / f"{spec['key']}.npz"
        _write_chunk(path, spec, output)
        completed[spec["key"]] = {
            "g": spec["g"],
            "gamma": spec["gamma"],
            "g_index": spec["g_index"],
            "chunk": spec["chunk"],
            "size": spec["size"],
            "seed": spec["seed"],
            "noise_kind": output["noise_kind"],
            "noise_hash": output["noise_hash"],
            "seconds": output["seconds"],
            "file": path.name,
            "sha256": _sha256(path),
            "completed_at": time.time(),
        }
        manifest["status"] = "running"
        _atomic_write_json(manifest_path, manifest)
        label = f"g={spec['g']:g}" if spec["g"] is not None else f"gamma={spec['gamma']:g}"
        print(
            f"DONE {len(completed)}/{len(specs)} {spec['key']} "
            f"{label} n={spec['size']} {output['seconds']:.1f}s",
            flush=True,
        )

    if args.workers == 1:
        for spec in pending:
            record(spec, run_chunk(spec))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(run_chunk, spec): spec for spec in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())


def _run_reference(args, output_dir: Path) -> None:
    jobs = {}
    for g in args.g_values:
        for n_grid in args.reference_grids:
            for h in args.reference_h:
                for offset in (-h, 0.0, h):
                    point = float(g * np.exp(offset))
                    jobs[(round(point, 15), n_grid)] = point
    print(f"reference: {len(jobs)} stationary eigenproblems", flush=True)
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=min(args.workers, len(jobs)))
    started = perf_counter()
    try:
        handles = {
            key: pool.apply_async(_reference_point, (point, key[1], args.J, args.dt))
            for key, point in jobs.items()
        }
        values = {}
        for key, handle in handles.items():
            remaining = max(1.0, args.reference_timeout - (perf_counter() - started))
            try:
                values[key] = ("ok", float(handle.get(timeout=remaining)))
            except Exception as error:  # timeout or eigensolver failure
                values[key] = (type(error).__name__, str(error))
    finally:
        pool.terminate()
        pool.join()

    points = []
    for g in args.g_values:
        combinations = []
        for n_grid in args.reference_grids:
            for h in args.reference_h:
                keys = [(round(float(g * np.exp(o)), 15), n_grid) for o in (-h, 0.0, h)]
                statuses = [values[key][0] for key in keys]
                entry = {"n_grid": n_grid, "h": h, "statuses": statuses}
                if all(status == "ok" for status in statuses):
                    minus, centre, plus = (values[key][1] for key in keys)
                    entry.update(
                        {
                            "q": centre,
                            "dq_dlog_g": (plus - minus) / (2.0 * h),
                            "d2q_dlog_g2": (plus - 2.0 * centre + minus) / (h * h),
                        }
                    )
                else:
                    entry["errors"] = [values[key][1] for key in keys if values[key][0] != "ok"]
                combinations.append(entry)
        good = [entry for entry in combinations if "q" in entry]
        usable = len(good) == len(combinations) and len(good) > 1
        if usable:
            for field in ("q", "dq_dlog_g", "d2q_dlog_g2"):
                values_field = np.array([entry[field] for entry in good])
                spread = float(np.ptp(values_field))
                scale = float(np.max(np.abs(values_field)))
                usable &= spread <= args.reference_atol + args.reference_rtol * scale
        finest = min(good, key=lambda e: (-e["n_grid"], e["h"])) if good else None
        points.append(
            {
                "g": float(g),
                "usable": bool(usable),
                "preferred": finest,
                "combinations": combinations,
            }
        )
    payload = {
        "description": (
            "Stationary single-qubit q and its log(g) derivatives from the "
            "deterministic transfer operator (3-point stencil on a noise-free "
            "curve). Usable only if all grid/step combinations agree within "
            "atol + rtol * scale."
        ),
        "J": args.J,
        "dt": args.dt,
        "rtol": args.reference_rtol,
        "atol": args.reference_atol,
        "timeout_seconds": args.reference_timeout,
        "points": points,
    }
    _atomic_write_json(output_dir / "reference.json", payload)
    for point in points:
        preferred = point["preferred"]
        summary = (
            f"q={preferred['q']:.5f} q'={preferred['dq_dlog_g']:.4f} "
            f"q''={preferred['d2q_dlog_g2']:.4f}"
            if preferred
            else "no successful combination"
        )
        print(f"reference g={point['g']:g} usable={point['usable']} {summary}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    n_burnin, n_samples, _ = _validate(args)
    output_dir = args.output_dir
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    configuration = _configuration(args, n_burnin, n_samples)
    fingerprint = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError("output directory holds a run with a different configuration")
    else:
        manifest = {
            "configuration": configuration,
            "fingerprint": fingerprint,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "git": _git_metadata(),
                "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "chunks": {},
            "status": "started",
        }
        _atomic_write_json(manifest_path, manifest)

    specs = _chunk_specs(args, n_burnin, n_samples)
    # Progress metadata for monitors; it does not enter the fingerprint.
    manifest["expected_chunks"] = len(specs)
    manifest["expected_trajectories"] = int(sum(spec["size"] for spec in specs))
    manifest["pid"] = os.getpid()
    manifest.setdefault("runs", []).append({"pid": os.getpid(), "started_at": time.time()})
    manifest["status"] = "running"
    manifest.pop("error", None)
    _atomic_write_json(manifest_path, manifest)

    started = perf_counter()
    try:
        _run_chunks(args, specs, chunk_dir, manifest, manifest_path)
        print(f"STUDY_COMPLETE chunks={len(specs)} seconds={perf_counter() - started:.1f}", flush=True)
        if args.reference:
            manifest["status"] = "reference"
            _atomic_write_json(manifest_path, manifest)
            _run_reference(args, output_dir)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _atomic_write_json(manifest_path, manifest)
        raise
    manifest["status"] = "complete"
    _atomic_write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
