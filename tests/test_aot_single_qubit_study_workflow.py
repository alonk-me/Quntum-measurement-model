from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPOSITORY_ROOT / "scripts" / "run_aot_single_qubit_study.py"
MONITOR = REPOSITORY_ROOT / "scripts" / "monitor_aot_single_qubit_study.py"


def _study_command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(DRIVER),
        "--output-dir",
        str(output_dir),
        "--gamma-values",
        "1.0",
        "--dt-values",
        "0.01",
        "--h-values",
        "0.01",
        "--n-batches",
        "2",
        "--n-trajectories",
        "2",
        "--n-burnin",
        "2",
        "--n-samples",
        "8",
        "--seed",
        "19",
    ]


def test_checkpointed_study_writes_and_resumes(tmp_path):
    first = subprocess.run(
        _study_command(tmp_path),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "STUDY_COMPLETE completed=2/2" in first.stdout

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["completed"]) == 2
    assert (tmp_path / "step_size_summary.csv").exists()
    assert (tmp_path / "records.jsonl").exists()
    with (tmp_path / "aggregate_summary.csv").open(newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert len(aggregate_rows) == 1
    assert aggregate_rows[0]["n_batches"] == "2"

    second = subprocess.run(
        _study_command(tmp_path),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SKIP 1/2 gamma=1|dt=0.01|h=0.01|batch=0" in second.stdout
    assert "SKIP 2/2 gamma=1|dt=0.01|h=0.01|batch=1" in second.stdout
    assert (tmp_path / "step_size_summary.csv").read_text().count("\n") == 3


def test_monitor_reads_completed_study(tmp_path):
    subprocess.run(
        _study_command(tmp_path),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            sys.executable,
            str(MONITOR),
            "--output-dir",
            str(tmp_path),
            "status",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "study_status=complete completed=2/2" in status.stdout
    assert "freshness=ok" in status.stdout

    plot = subprocess.run(
        [
            sys.executable,
            str(MONITOR),
            "--output-dir",
            str(tmp_path),
            "plot",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "live_progress.png" in plot.stdout
    assert (tmp_path / "live_progress.png").exists()
    snapshot = json.loads((tmp_path / "live_progress.json").read_text())
    assert snapshot["status"] == "COMPLETE"
    assert snapshot["completed_batches"] == 2


def test_guarded_profile_persists_and_reuses_transition(tmp_path):
    command = _study_command(tmp_path) + [
        "--profile",
        "tangent_guarded_v1",
        "--profile-max-second-tangent-norm",
        "0.000001",
    ]
    first = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "STUDY_COMPLETE completed=2/2" in first.stdout

    decisions = [
        json.loads(line)
        for line in (tmp_path / "profile_decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(decisions) == 2
    assert decisions[0]["transition"] in {"refined_dt", "shortened_horizon"}
    assert decisions[1]["schedule"]["fingerprint"] != decisions[0]["schedule"]["fingerprint"]

    second = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "STUDY_COMPLETE completed=2/2" in second.stdout
    assert len((tmp_path / "profile_decisions.jsonl").read_text().splitlines()) == 2
