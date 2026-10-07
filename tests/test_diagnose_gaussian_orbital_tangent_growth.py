from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from quantum_measurement.jw_expansion.gaussian_orbital import (
    coarsen_standard_normal_noise,
    simulate_gaussian_orbital_chain_second_order,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "diagnose_gaussian_orbital_tangent_growth.py"
SEED = 23


def _run(output_dir: Path, *extra: str) -> dict:
    command = [
        sys.executable,
        str(SCRIPT),
        "--L", "2",
        "--g", "1",
        "--dt", "0.01",
        "--burn-in-time", "0.04",
        "--sample-time", "0.12",
        "--n-trajectories", "2",
        "--fd-trajectories", "1",
        "--exact-control-trajectories", "1",
        "--h-values", "0.02", "0.01",
        "--record-stride", "2",
        "--seed", str(SEED),
        "--output-dir", str(output_dir),
        *extra,
    ]
    subprocess.run(
        command, cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return json.loads((output_dir / "tangent_growth_results.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_runs_finite_difference_and_exact_control(tmp_path):
    payload = _run(tmp_path)

    control = payload["exact_control"]
    assert control["base_z_rms_error"] < 1.0e-12
    assert control["base_final_covariance_rms_error"] < 1.0e-12

    for sweep in (payload["primary_finite_difference"], control["finite_difference"]):
        records = sweep["h_records"]
        assert [record["h"] for record in records] == [0.02, 0.01]
        for record in records:
            np.testing.assert_allclose(
                record["finite_d2q"], record["direct_d2q"], rtol=0.0, atol=2.0e-6
            )
        for name in ("u_rms_error", "v_rms_error", "F2_rms_error"):
            assert records[0][name] / records[1][name] > 3.0

    primary = payload["primary"]
    assert primary["d2q_reconstruction_error"] < 1.0e-12
    alignment = primary["jump_alignment"]
    assert alignment["window_steps"] == 2
    assert len(alignment["spearman_max_abs_v_vs_max_abs_dz"]) == 2

    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert set(hashes) == {
        "primary_tangent_timeseries.csv",
        "exact_control_tangent_timeseries.csv",
    }
    for name, digest in hashes.items():
        assert _sha256(tmp_path / name) == digest


def test_coarsening_factor_replays_coupled_coarse_path(tmp_path):
    payload = _run(tmp_path, "--coarsening-factor", "2", "--skip-exact-control")

    configuration = payload["configuration"]
    assert configuration["coarsening_factor"] == 2
    assert configuration["fine_dt"] == 0.01
    assert configuration["dt"] == 0.02
    assert (configuration["n_burnin"], configuration["n_samples"]) == (2, 6)

    fine_noise = np.random.default_rng(SEED).standard_normal((2, 16, 2))
    reference, _ = simulate_gaussian_orbital_chain_second_order(
        L=2,
        gamma=4.0,
        J=1.0,
        dt=0.02,
        n_burnin=2,
        n_samples=6,
        boundary="periodic",
        noise=coarsen_standard_normal_noise(fine_noise, 2),
    )
    assert configuration["noise_hash"] == reference.noise_hash
    np.testing.assert_array_equal(payload["primary"]["q"]["values"], reference.q)
    np.testing.assert_array_equal(
        payload["primary"]["d2q"]["values"], reference.d2q_dtheta2
    )
