from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from quantum_measurement.aot_single_qubit import (
    rademacher_noise,
    simulate_aot_second_order_cpu,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "diagnose_aot_single_qubit_tangent_spikes.py"


def test_single_qubit_spike_diagnostic_replays_and_converges(tmp_path):
    arguments = dict(gamma=1.0, J=1.0, dt=0.01, n_burnin=10, n_samples=50)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gamma", "1.0",
            "--J", "1.0",
            "--dt", "0.01",
            "--n-burnin", "10",
            "--n-samples", "50",
            "--n-trajectories", "3",
            "--seed", "31",
            "--h-values", "0.02", "0.01",
            "--top-k", "2",
            "--window-steps", "5",
            "--jump-window-steps", "5",
            "--output-dir", str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((tmp_path / "tangent_spike_results.json").read_text())

    reference = simulate_aot_second_order_cpu(
        **arguments, noise=rademacher_noise(3, 60, 31)
    )
    assert payload["configuration"]["noise_hash"] == reference.noise_hash
    np.testing.assert_array_equal(
        payload["direct"]["d2q"]["values"], reference.d2q_dtheta2
    )

    direct = payload["direct"]
    assert direct["d2q_reconstruction_error"] < 1.0e-12
    assert len(direct["jump_alignment"]["spearman_max_abs_v_vs_max_abs_dz"]) == 3

    coarse, fine = payload["finite_difference"]["h_records"]
    for name in ("u_rms_error", "v_rms_error"):
        assert coarse[name] / fine[name] > 3.0
    assert max(fine["d2q_abs_error"]) < 1.0e-5

    with (tmp_path / "top_trajectory_traces.csv").open(newline="") as handle:
        traces = list(csv.DictReader(handle))
    assert {int(row["trajectory"]) for row in traces} == set(direct["top_trajectories"])
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    digest = hashlib.sha256((tmp_path / "top_trajectory_traces.csv").read_bytes())
    assert hashes["top_trajectory_traces.csv"] == digest.hexdigest()
