from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from quantum_measurement.aot_single_qubit import (
    rademacher_noise,
    simulate_aot_second_order_cpu,
)
from quantum_measurement.jw_expansion.gaussian_orbital import (
    simulate_gaussian_orbital_chain_second_order,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_tangent_sample_scaling.py"
CHAIN = [
    "--system", "chain", "--L", "2", "--g-values", "0.5", "2",
    "--n-trajectories", "5", "--chunk-size", "2", "--first-chunk-size", "1",
    "--dt", "0.01", "--burn-in-time", "0.02", "--sample-time", "0.05",
    "--seed", "41",
]
SINGLE_QUBIT = [
    "--system", "single-qubit", "--g-values", "1",
    "--n-trajectories", "6", "--chunk-size", "4",
    "--dt", "0.01", "--burn-in-time", "0.05", "--sample-time", "0.1",
    "--seed", "43",
]


def _run(arguments: list[str], output_dir: Path, *extra: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments, "--output-dir", str(output_dir), *extra],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _load(output_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    return {
        key: dict(np.load(output_dir / "chunks" / entry["file"]))
        for key, entry in manifest["chunks"].items()
    }


def test_chain_chunks_equal_serial_calls_and_resume(tmp_path):
    output = _run(CHAIN, tmp_path)
    assert "STUDY_COMPLETE chunks=6" in output
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    sizes = {key: entry["size"] for key, entry in manifest["chunks"].items()}
    assert sizes == {
        "g0_c00000": 1, "g0_c00001": 2, "g0_c00002": 2,
        "g1_c00000": 1, "g1_c00001": 2, "g1_c00002": 2,
    }
    entry = manifest["chunks"]["g1_c00002"]
    assert entry["seed"] == 41 + 1000 * 2 + 2
    chunk = _load(tmp_path)["g1_c00002"]
    noise = np.random.default_rng(entry["seed"]).standard_normal((2, 7, 2))
    reference, _ = simulate_gaussian_orbital_chain_second_order(
        L=2, gamma=8.0, J=1.0, dt=0.01, n_burnin=2, n_samples=5,
        boundary="periodic", noise=noise,
    )
    np.testing.assert_array_equal(chunk["d2q_dtheta2"], reference.d2q_dtheta2)
    np.testing.assert_array_equal(chunk["q"], reference.q)
    assert entry["noise_hash"] == reference.noise_hash

    rerun = _run(CHAIN, tmp_path)
    assert "chunks: 6 complete, 0 pending" in rerun
    assert "DONE" not in rerun


def test_parallel_workers_match_serial(tmp_path):
    _run(CHAIN, tmp_path / "serial")
    _run(CHAIN, tmp_path / "parallel", "--workers", "2")
    serial = _load(tmp_path / "serial")
    parallel = _load(tmp_path / "parallel")
    assert serial.keys() == parallel.keys()
    for key in serial:
        for field, values in serial[key].items():
            np.testing.assert_array_equal(values, parallel[key][field])


def test_single_qubit_chunks_and_reference(tmp_path):
    output = _run(
        SINGLE_QUBIT, tmp_path,
        "--reference", "--reference-grids", "128", "256",
        "--reference-h", "0.2", "0.1", "--workers", "2",
    )
    assert "STUDY_COMPLETE chunks=2" in output
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["chunks"]["g0_c00001"]
    assert (entry["size"], entry["seed"]) == (2, 43 + 1000 + 1)
    chunk = _load(tmp_path)["g0_c00001"]
    reference = simulate_aot_second_order_cpu(
        gamma=4.0, J=1.0, dt=0.01, n_burnin=5, n_samples=10,
        noise=rademacher_noise(2, 15, entry["seed"]),
    )
    np.testing.assert_array_equal(chunk["d2q_dtheta2"], reference.d2q_dtheta2)

    stationary = json.loads((tmp_path / "reference.json").read_text())
    point = stationary["points"][0]
    assert point["g"] == 1.0
    assert len(point["combinations"]) == 4
    preferred = point["preferred"]
    assert preferred["n_grid"] == 256 and preferred["h"] == 0.1
    assert 1.0 <= preferred["q"] <= 2.0
    assert np.isfinite(preferred["d2q_dlog_g2"])
    assert isinstance(point["usable"], bool)


NO_DRIVE = [
    "--system", "single-qubit", "--J", "0", "--gamma-values", "0.5", "2",
    "--n-trajectories", "5", "--chunk-size", "3",
    "--dt", "0.01", "--burn-in-time", "0", "--sample-time", "0.2",
    "--seed", "47", "--save-final-z",
]


def test_no_drive_gamma_values_store_final_z_and_progress_metadata(tmp_path):
    output = _run(NO_DRIVE, tmp_path)
    assert "STUDY_COMPLETE chunks=4" in output
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["expected_chunks"] == 4 and manifest["expected_trajectories"] == 10
    assert "g_values" not in manifest["configuration"]
    assert manifest["configuration"]["gamma_values"] == [0.5, 2.0]
    entry = manifest["chunks"]["g1_c00001"]
    assert entry["g"] is None and entry["gamma"] == 2.0 and entry["size"] == 2
    assert entry["seed"] == 47 + 1000 * 2 + 1 and "completed_at" in entry
    chunk = _load(tmp_path)["g1_c00001"]
    reference = simulate_aot_second_order_cpu(
        gamma=2.0, J=0.0, dt=0.01, n_burnin=0, n_samples=20,
        noise=rademacher_noise(2, 20, entry["seed"]),
    )
    for field in ("q", "dq_dtheta", "d2q_dtheta2", "final_z"):
        np.testing.assert_array_equal(chunk[field], getattr(reference, field))


def test_parameter_and_drive_arguments_are_validated(tmp_path):
    base = ["--system", "single-qubit", "--n-trajectories", "4", "--dt", "0.01",
            "--burn-in-time", "0", "--sample-time", "0.1", "--output-dir", str(tmp_path)]
    for extra in (
        ["--g-values", "1", "--gamma-values", "4"],
        [],
        ["--J", "0", "--g-values", "1"],
        ["--J", "0", "--gamma-values", "1", "--reference"],
    ):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *base, *extra],
            cwd=REPOSITORY_ROOT, capture_output=True, text=True,
        )
        assert completed.returncode != 0, extra


def test_monitor_reports_progress_and_writes_dashboard(tmp_path):
    job = tmp_path / "no_drive"
    _run(NO_DRIVE, job)
    (tmp_path / "jobs.json").write_text(json.dumps([
        {"name": "no_drive", "dir": "no_drive", "estimate_minutes": 1},
        {"name": "not_started", "dir": "not_started", "estimate_minutes": 2},
    ]))
    monitor = REPOSITORY_ROOT / "scripts" / "monitor_tangent_sample_scaling.py"
    status = subprocess.run(
        [sys.executable, str(monitor), "--root", str(tmp_path), "status"],
        cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    no_drive = next(line for line in status.splitlines() if line.startswith("no_drive"))
    assert "complete" in no_drive and "4/4" in no_drive and "10/10" in no_drive and "100.0" in no_drive
    assert "queued" in next(line for line in status.splitlines() if line.startswith("not_started"))
    assert "jobs complete 1/2" in status
    subprocess.run(
        [sys.executable, str(monitor), "--root", str(tmp_path), "plot"],
        cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
    )
    assert (tmp_path / "live_progress.png").stat().st_size > 0
