#!/usr/bin/env python3
"""Compare COGS B0 vs B1 gen exact match by category.

Reads the most recent run for each variant from runs/cogs/.
Uses greedy_gen_by_cat from summary.json if available,
otherwise falls back to the last epoch's gen.by_cat from metrics.json.

Usage:
  python3 cogs_compare.py
  python3 cogs_compare.py --b0-dir runs/cogs/B0_s42_XXXX --b1-dir runs/cogs/B1_s42_XXXX
"""
import os, json, argparse, glob

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "cogs")


def find_latest_run(variant):
    pattern = os.path.join(RUNS_DIR, f"{variant}_s*")
    runs = sorted(glob.glob(pattern))
    return runs[-1] if runs else None


def load_by_cat(run_dir):
    """Try summary.json greedy first, then metrics.json last epoch."""
    s = os.path.join(run_dir, "summary.json")
    if os.path.exists(s):
        d = json.load(open(s))
        if d.get("greedy_gen_by_cat"):
            return d["greedy_gen_by_cat"], "greedy"
    m = os.path.join(run_dir, "metrics.json")
    if os.path.exists(m):
        metrics = json.load(open(m))
        if metrics:
            last = metrics[-1]
            if last.get("gen", {}).get("by_cat"):
                return last["gen"]["by_cat"], f"tf_ep{last['epoch']}"
    return {}, "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b0-dir", type=str, default=None)
    p.add_argument("--b1-dir", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default=None,
                   help="Override runs directory")
    args = p.parse_args()
    global RUNS_DIR
    if args.runs_dir:
        RUNS_DIR = args.runs_dir

    b0_dir = args.b0_dir or find_latest_run("B0")
    b1_dir = args.b1_dir or find_latest_run("B1")

    if not b0_dir or not os.path.isdir(b0_dir):
        print(f"B0 run not found (looked in {RUNS_DIR}/B0_s*)")
        return
    if not b1_dir or not os.path.isdir(b1_dir):
        print(f"B1 run not found (looked in {RUNS_DIR}/B1_s*)")
        return

    b0_cats, b0_src = load_by_cat(b0_dir)
    b1_cats, b1_src = load_by_cat(b1_dir)

    all_cats = sorted(set(list(b0_cats.keys()) + list(b1_cats.keys())))
    if not all_cats:
        print("No per-category data found. Runs may need to finish or be re-run with by_cat logging.")
        return

    print(f"{'='*80}")
    print(f"  COGS gen exact match — B0 vs B1 by category")
    print(f"{'='*80}")
    print(f"  B0: {os.path.basename(b0_dir)} ({b0_src})")
    print(f"  B1: {os.path.basename(b1_dir)} ({b1_src})")
    print()
    print(f"  {'Category':<55s} {'B0':>7s} {'B1':>7s} {'Δ':>7s}")
    print(f"  {'-'*55} {'-'*7} {'-'*7} {'-'*7}")

    b0_total, b1_total = 0.0, 0.0
    n_cats = 0
    for cat in all_cats:
        v0 = b0_cats.get(cat, 0.0)
        v1 = b1_cats.get(cat, 0.0)
        delta = v1 - v0
        sign = "+" if delta > 0 else ""
        print(f"  {cat:<55s} {v0:6.2f}% {v1:6.2f}% {sign}{delta:6.2f}%")
        b0_total += v0; b1_total += v1; n_cats += 1

    print(f"  {'-'*55} {'-'*7} {'-'*7} {'-'*7}")
    b0_avg = b0_total / max(n_cats, 1)
    b1_avg = b1_total / max(n_cats, 1)
    d_avg = b1_avg - b0_avg
    sign = "+" if d_avg > 0 else ""
    print(f"  {'AVERAGE':<55s} {b0_avg:6.2f}% {b1_avg:6.2f}% {sign}{d_avg:6.2f}%")

    # Save
    out = {
        "b0_dir": b0_dir, "b1_dir": b1_dir,
        "b0_src": b0_src, "b1_src": b1_src,
        "categories": {cat: {"B0": b0_cats.get(cat, 0.0),
                             "B1": b1_cats.get(cat, 0.0),
                             "delta": b1_cats.get(cat, 0.0) - b0_cats.get(cat, 0.0)}
                       for cat in all_cats},
        "avg_B0": b0_avg, "avg_B1": b1_avg, "avg_delta": d_avg,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogs_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
