#!/usr/bin/env python3
"""Render a live dashboard for a checkpointed single-qubit study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _alive(pid: object) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _status(output_dir: Path, manifest: dict[str, object], configuration: dict[str, object]) -> str:
    state = _read_json(output_dir / "run_state.json", {})
    pid = state.get("pid") if isinstance(state, dict) else None
    completed = len(manifest.get("completed", []))
    config = configuration.get("configuration", {})
    expected = (
        len(config.get("gamma_values", []))
        * len(config.get("dt_values", []))
        * len(config.get("h_values", []))
        * int(config.get("n_batches", 1))
    ) if isinstance(config, dict) else 0
    last_update = manifest.get("last_update_unix")
    stale_after = float(state.get("stale_after_seconds", 900.0)) if isinstance(state, dict) else 900.0
    stale = last_update is not None and time.time() - float(last_update) > stale_after and _alive(pid)
    if stale:
        return "STALE"
    if manifest.get("status") == "complete":
        return "COMPLETE"
    if completed == 0 and not _alive(pid):
        return "BLOCKED"
    if not _alive(pid) and completed < expected:
        return "FAILED"
    return "RUNNING"


def _number(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def render(output_dir: Path, output_path: Path) -> None:
    manifest = _read_json(output_dir / "manifest.json", {})
    configuration = _read_json(output_dir / "configuration.json", {})
    rows = _read_rows(output_dir / "aggregate_summary.csv")
    state = _status(output_dir, manifest, configuration)
    config = configuration.get("configuration", {}) if isinstance(configuration, dict) else {}
    completed = len(manifest.get("completed", [])) if isinstance(manifest, dict) else 0
    expected = (
        len(config.get("gamma_values", []))
        * len(config.get("dt_values", []))
        * len(config.get("h_values", []))
        * int(config.get("n_batches", 1))
    ) if isinstance(config, dict) else 0

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    figure.suptitle(f"Single-qubit derivative study: {state}", fontsize=15, fontweight="bold")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    progress = axes[0, 0]
    progress.set_axis_off()
    fraction = completed / expected if expected else 0.0
    progress.barh(0, fraction, color="tab:green", height=0.25)
    progress.set_xlim(0, 1)
    progress.set_ylim(-0.6, 0.8)
    progress.text(0.02, 0.58, f"{completed}/{expected} batches", transform=progress.transAxes, fontsize=14)
    progress.text(0.02, 0.42, f"{100.0 * fraction:.1f}% complete", transform=progress.transAxes)
    progress.text(0.02, 0.28, f"aggregate points: {len(rows)}", transform=progress.transAxes)
    progress.text(0.02, 0.14, f"last update: {manifest.get('last_completed', '-')}", transform=progress.transAxes, fontsize=8)
    progress.set_title("Progress")

    if rows:
        min_h = min(float(row["h"]) for row in rows)
        selected = [row for row in rows if float(row["h"]) == min_h]
        selected.sort(key=lambda row: float(row["gamma"]))
        gamma = _number(selected, "gamma")

        axes[0, 1].errorbar(gamma, _number(selected, "dq_mean"), yerr=_number(selected, "dq_sem"), marker="o", capsize=3)
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_title(f"First derivative, h={min_h:g}")
        axes[0, 1].set_xlabel("gamma")
        axes[0, 1].set_ylabel("dq/dtheta")

        axes[0, 2].errorbar(gamma, _number(selected, "d2q_mean"), yerr=_number(selected, "d2q_sem"), marker="o", capsize=3, color="tab:orange")
        axes[0, 2].set_xscale("log")
        axes[0, 2].set_title(f"Second derivative, h={min_h:g}")
        axes[0, 2].set_xlabel("gamma")
        axes[0, 2].set_ylabel("d2q/dtheta2")

        axes[1, 0].plot(gamma, _number(selected, "dq_rms_error"), "o-", label="dq")
        axes[1, 0].plot(gamma, _number(selected, "d2q_rms_error"), "s-", label="d2q")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("Tangent vs finite difference")
        axes[1, 0].set_xlabel("gamma")
        axes[1, 0].set_ylabel("RMS error")
        axes[1, 0].legend()

        axes[1, 1].plot(gamma, _number(selected, "max_first_tangent_norm"), "o-", label="max |eta|")
        axes[1, 1].plot(gamma, _number(selected, "max_second_tangent_norm"), "s-", label="max |phi|")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Tangent growth")
        axes[1, 1].set_xlabel("gamma")
        axes[1, 1].set_ylabel("maximum norm")
        axes[1, 1].legend()

        axes[1, 2].plot(gamma, _number(selected, "tangent_seconds"), "o-", label="tangent")
        axes[1, 2].plot(gamma, _number(selected, "finite_difference_seconds"), "s-", label="finite difference")
        axes[1, 2].set_xscale("log")
        axes[1, 2].set_title("Runtime per batch")
        axes[1, 2].set_xlabel("gamma")
        axes[1, 2].set_ylabel("seconds")
        axes[1, 2].legend()
    else:
        for axis in axes.flat[1:]:
            axis.text(0.5, 0.5, "Waiting for first completed batch", ha="center", va="center")

    figure.tight_layout()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    figure.savefig(temporary, dpi=140, format="png")
    plt.close(figure)
    temporary.replace(output_path)

    snapshot = {
        "status": state,
        "completed_batches": completed,
        "expected_batches": expected,
        "aggregate_points": len(rows),
        "last_completed": manifest.get("last_completed"),
        "updated_unix": time.time(),
        "configuration_fingerprint": configuration.get("fingerprint"),
    }
    snapshot_path = output_path.with_suffix(".json")
    snapshot_tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    snapshot_tmp.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    snapshot_tmp.replace(snapshot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.output_dir / "live_progress.png"
    render(args.output_dir, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
