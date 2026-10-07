#!/usr/bin/env python3
"""Read-only progress monitor for ``run_tangent_sample_scaling.py`` jobs.

``--root`` is either one run directory (holding ``manifest.json``) or a
directory of runs.  If it holds ``jobs.json`` (written by
``run_derivative_overview.sh``), jobs are listed in that order and jobs not yet
started appear as queued.  The monitor never modifies run outputs.

Commands:
  status            one table: state, progress, throughput, elapsed, ETA
  watch             refresh the table every ``--interval`` seconds
  plot              write ``live_progress.png`` (progress bars + live q preview)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--plot", action="store_true", help="with watch: also refresh the dashboard")
    parser.add_argument("--output", type=Path, default=None, help="dashboard path (default: ROOT/live_progress.png)")
    parser.add_argument("command", choices=("status", "watch", "plot"))
    return parser.parse_args(argv)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _jobs(root: Path) -> list[dict]:
    """Return ``[{name, dir, estimate_minutes}]`` in display order."""

    plan = _read_json(root / "jobs.json")
    if isinstance(plan, list):
        return [
            {"name": job["name"], "dir": root / job["dir"], "estimate_minutes": job.get("estimate_minutes")}
            for job in plan
        ]
    if (root / "manifest.json").exists():
        return [{"name": root.name, "dir": root, "estimate_minutes": None}]
    return [
        {"name": path.parent.name, "dir": path.parent, "estimate_minutes": None}
        for path in sorted(root.glob("*/manifest.json"))
    ]


def _last_exit_code(directory: Path):
    log = directory / "driver.log"
    if not log.exists():
        return None
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("exit="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def job_status(job: dict, now: float) -> dict:
    """Summarize one job from its manifest; never writes anything."""

    directory = job["dir"]
    manifest = _read_json(directory / "manifest.json")
    row = {
        "name": job["name"], "state": "queued", "done_chunks": 0, "expected_chunks": None,
        "done_traj": 0, "expected_traj": None, "percent": 0.0, "rate": None, "elapsed": None,
        "eta": None, "age": None, "alive": False, "estimate_minutes": job.get("estimate_minutes"),
        "error": None,
    }
    exit_code = _last_exit_code(directory)
    if manifest is None:
        if exit_code not in (None, 0):
            row["state"] = "failed"
            row["error"] = f"exit={exit_code} before writing a manifest (see driver.log)"
        return row

    chunks = manifest.get("chunks", {})
    row["done_chunks"] = len(chunks)
    row["done_traj"] = int(sum(entry["size"] for entry in chunks.values()))
    row["expected_chunks"] = manifest.get("expected_chunks")
    row["expected_traj"] = manifest.get("expected_trajectories")
    if row["expected_traj"]:
        row["percent"] = 100.0 * row["done_traj"] / row["expected_traj"]
    row["alive"] = _alive(manifest.get("pid"))
    row["error"] = manifest.get("error")

    runs = manifest.get("runs") or []
    started = runs[-1]["started_at"] if runs else None
    times = sorted(entry["completed_at"] for entry in chunks.values() if "completed_at" in entry)
    last_update = max(times) if times else started
    if last_update is not None:
        row["age"] = now - last_update
    if started is not None:
        current = [entry for entry in chunks.values() if entry.get("completed_at", 0) >= started]
        end = now if manifest.get("status") in ("running", "started", "reference") else (last_update or now)
        row["elapsed"] = max(end - started, 1e-9)
        done_now = sum(entry["size"] for entry in current)
        if done_now:
            row["rate"] = done_now / row["elapsed"]
            if row["expected_traj"]:
                row["eta"] = (row["expected_traj"] - row["done_traj"]) / row["rate"]

    status = manifest.get("status", "started")
    if status == "complete":
        row["state"] = "complete"
        row["eta"] = 0.0
    elif status == "failed" or (exit_code not in (None, 0)):
        row["state"] = "failed"
    elif not row["alive"]:
        row["state"] = "dead" if exit_code is None else "stopped"
    else:
        row["state"] = "reference" if status == "reference" else "running"
        intervals = np.diff(times) if len(times) > 2 else np.array([])
        typical = float(np.median(intervals)) if intervals.size else None
        if status != "reference" and typical and row["age"] is not None and row["age"] > max(60.0, 3.0 * typical):
            row["state"] = "stale"
    return row


def _duration(seconds) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def status_table(root: Path, now=None) -> tuple[str, list[dict]]:
    now = time.time() if now is None else now
    rows = [job_status(job, now) for job in _jobs(root)]
    header = (f"{'job':24s} {'state':9s} {'chunks':>11s} {'trajectories':>19s} {'%':>6s} "
              f"{'traj/s':>8s} {'elapsed':>8s} {'ETA':>8s} {'upd age':>8s} {'alive':>5s}")
    lines = [header, "-" * len(header)]
    for row in rows:
        chunks = f"{row['done_chunks']}/{row['expected_chunks'] or '?'}"
        traj = f"{row['done_traj']}/{row['expected_traj'] or '?'}"
        rate = f"{row['rate']:.1f}" if row["rate"] else "-"
        lines.append(
            f"{row['name'][:24]:24s} {row['state']:9s} {chunks:>11s} {traj:>19s} {row['percent']:6.1f} "
            f"{rate:>8s} {_duration(row['elapsed']):>8s} {_duration(row['eta']):>8s} "
            f"{_duration(row['age']):>8s} {'yes' if row['alive'] else 'no':>5s}"
        )
        if row["error"]:
            lines.append(f"{'':24s} error: {row['error']}")
    remaining = 0.0
    unknown = False
    for row in rows:
        if row["state"] in ("running", "stale", "reference"):
            if row["eta"] is None:
                unknown = True
            else:
                remaining += row["eta"]
        elif row["state"] == "queued":
            if row["estimate_minutes"] is None:
                unknown = True
            else:
                remaining += 60.0 * row["estimate_minutes"]
    complete = sum(row["state"] == "complete" for row in rows)
    eta = "unknown" if unknown and remaining == 0 else _duration(remaining) + (" (+unknown)" if unknown else "")
    lines.append("-" * len(header))
    lines.append(f"jobs complete {complete}/{len(rows)}; overall ETA {eta}; "
                 f"checked {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}")
    return "\n".join(lines), rows


def _preview_points(directory: Path):
    """Return parameter values and q mean/CI for points with completed chunks."""

    manifest = _read_json(directory / "manifest.json")
    if not manifest:
        return None
    config = manifest["configuration"]
    if "g_values" in config:
        values, label = np.log10(config["g_values"]), "log10 g"
    else:
        values, label = np.log(config["gamma_values"]), "ln gamma"
    sums = {}
    for entry in manifest.get("chunks", {}).values():
        path = directory / "chunks" / entry["file"]
        try:
            q = np.load(path)["q"]
        except (OSError, KeyError, ValueError):
            continue
        accumulator = sums.setdefault(entry["g_index"], [0, 0.0, 0.0])
        accumulator[0] += q.size
        accumulator[1] += float(np.sum(q))
        accumulator[2] += float(np.sum(q * q))
    points = []
    for index, (count, total, squares) in sorted(sums.items()):
        mean = total / count
        sem = np.sqrt(max(squares / count - mean * mean, 0.0) / max(count - 1, 1)) if count > 1 else 0.0
        points.append((values[index], mean, 1.96 * sem, count))
    return label, points


def plot(root: Path, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text, rows = status_table(root)
    jobs = _jobs(root)
    n = len(jobs)
    columns = min(3, max(1, n))
    preview_rows = int(np.ceil(n / columns))
    fig = plt.figure(figsize=(5.2 * columns, 2.2 + 0.45 * n + 3.2 * preview_rows))
    grid = fig.add_gridspec(1 + preview_rows, columns, height_ratios=[0.6 + 0.45 * n] + [3.2] * preview_rows)

    ax = fig.add_subplot(grid[0, :])
    colors = {"complete": "tab:green", "running": "tab:blue", "reference": "tab:purple", "stale": "tab:orange",
              "failed": "tab:red", "dead": "tab:red", "stopped": "tab:red", "queued": "lightgray"}
    for i, row in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, 100, color="whitesmoke", edgecolor="lightgray")
        ax.barh(y, row["percent"], color=colors.get(row["state"], "gray"))
        detail = f"{row['state']}  {row['percent']:.0f}%  ETA {_duration(row['eta'])}"
        ax.text(1, y, f"{row['name']}: {detail}", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("percent of trajectories complete")
    ax.set_title(text.splitlines()[-1], fontsize=9)

    for i, job in enumerate(jobs):
        ax = fig.add_subplot(grid[1 + i // columns, i % columns])
        preview = _preview_points(job["dir"])
        ax.set_title(f"{job['name']}: live q preview", fontsize=9)
        if not preview or not preview[1]:
            ax.text(0.5, 0.5, "no completed chunks yet", ha="center", va="center", transform=ax.transAxes)
            continue
        label, points = preview
        x, mean, half, count = map(np.array, zip(*points))
        ax.errorbar(x, mean, yerr=half, fmt="o", ms=3, capsize=2)
        ax.set_xlabel(label)
        ax.set_ylabel("mean q (95% CI)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.png")
    fig.savefig(temporary, dpi=110)
    plt.close(fig)
    temporary.replace(output)
    return output


def main(argv=None) -> None:
    args = parse_args(argv)
    output = args.output or (args.root / "live_progress.png")
    if args.command == "status":
        print(status_table(args.root)[0])
    elif args.command == "plot":
        print(f"wrote {plot(args.root, output)}")
    else:
        try:
            while True:
                text, rows = status_table(args.root)
                sys.stdout.write("\033[2J\033[H" + text + "\n")
                if args.plot:
                    plot(args.root, output)
                    sys.stdout.write(f"dashboard: {output}\n")
                sys.stdout.flush()
                if all(row["state"] in ("complete", "failed", "dead", "stopped") for row in rows):
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
