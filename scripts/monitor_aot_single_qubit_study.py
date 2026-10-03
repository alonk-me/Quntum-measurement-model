#!/usr/bin/env python3
"""Start and monitor a checkpointed single-qubit derivative study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_STALE_SECONDS = 900.0


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "state": output_dir / "run_state.json",
        "manifest": output_dir / "manifest.json",
        "configuration": output_dir / "configuration.json",
        "rows": output_dir / "step_size_summary.csv",
        "log": output_dir / "driver.log",
    }


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _save_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _driver_path() -> Path:
    return Path(__file__).with_name("run_aot_single_qubit_study.py")


def _start(output_dir: Path, driver_args: list[str], resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_dir)
    state = _read_json(paths["state"], {})
    previous_pid = state.get("pid") if isinstance(state, dict) else None
    if _alive(int(previous_pid)) if previous_pid else False:
        raise RuntimeError(f"study already running with pid {previous_pid}")

    if resume and isinstance(state, dict) and state.get("command"):
        command = [str(value) for value in state["command"]]
    else:
        command = [
            sys.executable,
            "-u",
            str(_driver_path()),
            "--output-dir",
            str(output_dir),
            *driver_args,
        ]
    log_handle = paths["log"].open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=_driver_path().parents[1],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    _save_state(
        paths["state"],
        {
            "pid": process.pid,
            "started_unix": time.time(),
            "command": command,
            "output_dir": str(output_dir),
            "stale_after_seconds": DEFAULT_STALE_SECONDS,
        },
    )
    print(f"started pid={process.pid}")
    print(f"log={paths['log']}")
    print(f"status=python {Path(__file__).name} status --output-dir {output_dir}")


def _status(output_dir: Path, stale_after: float) -> None:
    paths = _paths(output_dir)
    state = _read_json(paths["state"], {})
    manifest = _read_json(paths["manifest"], {})
    configuration = _read_json(paths["configuration"], {})
    pid = state.get("pid") if isinstance(state, dict) else None
    alive = _alive(int(pid)) if pid else False
    completed = manifest.get("completed", []) if isinstance(manifest, dict) else []
    config = configuration.get("configuration", {}) if isinstance(configuration, dict) else {}
    expected = (
        len(config.get("gamma_values", []))
        * len(config.get("dt_values", []))
        * len(config.get("h_values", []))
        * int(config.get("n_batches", 1))
        if isinstance(config, dict)
        else 0
    )
    last_update = manifest.get("last_update_unix") if isinstance(manifest, dict) else None
    age = time.time() - float(last_update) if last_update else None
    stale = bool(alive and age is not None and age > stale_after)
    status = manifest.get("status", "not_started") if isinstance(manifest, dict) else "not_started"
    print(f"process={'alive' if alive else 'not_alive'} pid={pid or '-'}")
    print(f"study_status={status} completed={len(completed)}/{expected}")
    print(f"last_completed={manifest.get('last_completed', '-') if isinstance(manifest, dict) else '-'}")
    print(f"last_update_age_seconds={age:.1f}" if age is not None else "last_update_age_seconds=-")
    print(f"freshness={'STALE' if stale else 'ok'}")
    print(f"log={paths['log']}")
    if stale:
        print("action=inspect logs, then stop or resume after diagnosing the stale block")


def _logs(output_dir: Path, lines: int) -> None:
    path = _paths(output_dir)["log"]
    if not path.exists():
        print(f"no log at {path}")
        return
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


def _view(output_dir: Path) -> None:
    path = _paths(output_dir)["rows"]
    if not path.exists():
        print(f"no rows at {path}")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"rows={len(rows)}")
    for row in rows[-3:]:
        print(
            f"gamma={row['gamma']} dt={row['dt']} h={row['h']} "
            f"dq={row['dq_mean']} d2q={row['d2q_mean']} "
            f"dq_rms={row['dq_rms_error']} d2q_rms={row['d2q_rms_error']}"
        )


def _plot(output_dir: Path) -> None:
    script = Path(__file__).with_name("plot_aot_single_qubit_study.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        cwd=script.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip())


def _stop(output_dir: Path) -> None:
    paths = _paths(output_dir)
    state = _read_json(paths["state"], {})
    pid = state.get("pid") if isinstance(state, dict) else None
    if not pid or not _alive(int(pid)):
        print("study process is not running")
        return
    os.kill(int(pid), signal.SIGTERM)
    print(f"sent SIGTERM to pid={pid}")


def _watch(output_dir: Path, interval: float, stale_after: float) -> None:
    while True:
        print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        _status(output_dir, stale_after)
        _plot(output_dir)
        state = _read_json(_paths(output_dir)["state"], {})
        pid = state.get("pid") if isinstance(state, dict) else None
        manifest = _read_json(_paths(output_dir)["manifest"], {})
        if (not pid or not _alive(int(pid))) and isinstance(manifest, dict) and manifest.get("status") in {"complete", "partial"}:
            return
        time.sleep(max(interval, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "resume", "status", "logs", "view", "plot", "stop", "watch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lines", type=int, default=40)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--stale-after", type=float, default=DEFAULT_STALE_SECONDS)
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    driver_args = args.driver_args[1:] if args.driver_args[:1] == ["--"] else args.driver_args
    if args.command == "start":
        _start(args.output_dir, driver_args, resume=False)
    elif args.command == "resume":
        _start(args.output_dir, driver_args, resume=True)
    elif args.command == "status":
        _status(args.output_dir, args.stale_after)
    elif args.command == "logs":
        _logs(args.output_dir, args.lines)
    elif args.command == "view":
        _view(args.output_dir)
    elif args.command == "plot":
        _plot(args.output_dir)
    elif args.command == "stop":
        _stop(args.output_dir)
    else:
        _watch(args.output_dir, args.interval, args.stale_after)


if __name__ == "__main__":
    main()
