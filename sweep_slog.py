#!/usr/bin/env python3
"""SLOG sweep orchestrator — Julien avril 2026.

Runs the 2-config × 3-seed sweep to answer: is the +1.6 gen greedy gain
from filtering unaccusative verbs out of the permutation pool robust
across seeds or is it noise?

Interleaves configs so that an interruption leaves paired results:
  A/s42 → B/s42 → A/s123 → B/s123 → A/s456 → B/s456

Usage:
  python3 sweep_slog.py
  python3 sweep_slog.py --configs A B --seeds 42 123 456
  python3 sweep_slog.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "runs_sweep_slog")
TIMING_PATH = os.path.join(SWEEP_DIR, "timing.json")


def _run_dir(config, seed):
    base = "A_baseline_pool130" if config == "A" else "B_pool120_filtered"
    return os.path.join(SWEEP_DIR, base, f"s{seed}")


def _build_cmd(config, seed, epochs):
    common = [
        sys.executable, os.path.join(HERE, "slog_compositional.py"),
        "--variant", "B4", "--copy",
        "--peel-stack", "--peel-stack-depth=10",
        "--peel-cp", "--peel-cp-depth=10",
        "--permute-verbs",
        "--wh-passive-aug",
        "--tfr-start=20",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--runs-dir", _run_dir(config, seed),
    ]
    if config == "B":
        common.insert(common.index("--wh-passive-aug") + 1,
                      "--filter-unaccusative-from-pool")
    return common


def _has_summary(config, seed):
    """A run is considered done if *any* summary.json lives under its runs-dir."""
    rd = _run_dir(config, seed)
    if not os.path.isdir(rd):
        return False
    for root, _dirs, files in os.walk(rd):
        if "summary.json" in files:
            return True
    return False


def _stream_subset(process, log_file, show_star_only=True):
    """Stream child stdout to log file; echo only the ★ lines (best epochs)
    to the console, plus any line that doesn't look like a per-epoch tick."""
    for raw in iter(process.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace")
        log_file.write(line)
        log_file.flush()
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        if show_star_only:
            if "★" in stripped or not stripped.lstrip().startswith("E0") \
                    and not stripped.lstrip().startswith("E1") \
                    and not stripped.lstrip().startswith("E2") \
                    and not stripped.lstrip().startswith("E3") \
                    and not stripped.lstrip().startswith("E4") \
                    and not stripped.lstrip().startswith("E5"):
                print(stripped, flush=True)
        else:
            print(stripped, flush=True)


def _load_timing():
    if os.path.exists(TIMING_PATH):
        with open(TIMING_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_timing(data):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    with open(TIMING_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["A", "B"],
                    help="Configs to run (A = baseline pool 130, B = pool 120 filtered).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                    help="Seeds to run, space-separated.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip runs where summary.json already exists (default).")
    ap.add_argument("--no-skip-existing", action="store_false",
                    dest="skip_existing")
    ap.add_argument("--resume", action="store_true",
                    help="Alias of --skip-existing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the 6 commands without running.")
    ap.add_argument("--skip-analysis", action="store_true",
                    help="Do not call analyze_sweep.py at the end.")
    args = ap.parse_args()
    if args.resume:
        args.skip_existing = True

    os.makedirs(SWEEP_DIR, exist_ok=True)
    # Interleaved order: (A, s42), (B, s42), (A, s123), (B, s123), ...
    order = [(c, s) for s in args.seeds for c in args.configs]

    if args.dry_run:
        for config, seed in order:
            cmd = _build_cmd(config, seed, args.epochs)
            print(" ".join(cmd))
        return

    timing = _load_timing()
    total = len(order)
    for idx, (config, seed) in enumerate(order, 1):
        run_dir = _run_dir(config, seed)
        if args.skip_existing and _has_summary(config, seed):
            print(f"[{idx}/{total}] SKIP {config}/s{seed} (summary.json exists)",
                  flush=True)
            continue
        os.makedirs(run_dir, exist_ok=True)
        cmd = _build_cmd(config, seed, args.epochs)
        log_path = os.path.join(run_dir, "stdout.log")
        print(f"\n[{idx}/{total}] RUN {config}/s{seed} → {run_dir}", flush=True)
        print(f"         log: {log_path}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            _stream_subset(proc, log_file, show_star_only=True)
            rc = proc.wait()
        dt = time.time() - t0
        key = f"{config}_s{seed}"
        timing[key] = {"seconds": round(dt, 1), "minutes": round(dt / 60, 2),
                       "returncode": rc}
        _save_timing(timing)
        if rc != 0:
            print(f"\n[{idx}/{total}] FAILED {config}/s{seed} rc={rc} — "
                  f"see {log_path}", flush=True)
            # Continue anyway — skip-existing will protect completed runs.

    print("\nAll runs done (or skipped).", flush=True)

    if not args.skip_analysis:
        print("\n=== Running analysis ===", flush=True)
        analyze = os.path.join(HERE, "analyze_sweep.py")
        if os.path.exists(analyze):
            rc = subprocess.call([sys.executable, analyze])
            if rc != 0:
                print(f"  analyze_sweep.py exited with rc={rc}", flush=True)
        else:
            print("  analyze_sweep.py not found — run it manually.", flush=True)


if __name__ == "__main__":
    main()
