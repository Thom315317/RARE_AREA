#!/usr/bin/env python3
"""Aggregate A4_permute runs (or any variant) across seeds.

Reads runs/scan_compositional/<variant>_s<seed>_*/metrics.json.
Emits results_summary.json with mean and std of final + best test exact match.

Usage:
  python3 results_summary.py                       # defaults to A4_permute
  python3 results_summary.py --variant A0          # or any variant
  python3 results_summary.py --variant A4_permute --seeds 42,123,456
"""
import os, json, argparse, glob
import numpy as np

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "runs", "scan_compositional")


def load_run_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def per_run(run_dir):
    m = load_run_metrics(run_dir)
    if not m:
        return None
    last = m[-1]
    return {
        "run_dir": run_dir,
        "epochs": len(m),
        "last_epoch": last["epoch"],
        "final_test_exact": last["test"]["exact"],
        "best_test_exact":  max(x["test"]["exact"] for x in m),
        "final_val_exact":  last["val"]["exact"],
        "best_val_exact":   max(x["val"]["exact"] for x in m),
        "final_train_loss": last["train_loss"],
    }


def aggregate(variant, seeds=None):
    pattern = os.path.join(RUNS_DIR, f"{variant}_s*_*")
    all_runs = sorted(glob.glob(pattern))
    rows = []
    for d in all_runs:
        seed = int(d.split(f"{variant}_s")[1].split("_")[0])
        if seeds and seed not in seeds:
            continue
        r = per_run(d)
        if r is None: continue
        r["seed"] = seed
        rows.append(r)
    # Keep only the most recent run per seed (largest timestamp)
    by_seed = {}
    for r in rows:
        if r["seed"] not in by_seed or r["run_dir"] > by_seed[r["seed"]]["run_dir"]:
            by_seed[r["seed"]] = r
    rows = sorted(by_seed.values(), key=lambda x: x["seed"])
    if not rows:
        return None
    finals = np.array([r["final_test_exact"] for r in rows])
    bests  = np.array([r["best_test_exact"]  for r in rows])
    return {
        "variant": variant,
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "final_test_mean": float(finals.mean()),
        "final_test_std":  float(finals.std(ddof=0)),
        "best_test_mean":  float(bests.mean()),
        "best_test_std":   float(bests.std(ddof=0)),
        "per_run": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="A4_permute")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seed list, e.g. 42,123,456")
    p.add_argument("--runs-dir", type=str, default=None,
                   help="Override runs directory to scan")
    p.add_argument("--out", type=str, default="results_summary.json")
    args = p.parse_args()
    global RUNS_DIR
    if args.runs_dir:
        RUNS_DIR = args.runs_dir
    seeds = None
    if args.seeds:
        seeds = set(int(s) for s in args.seeds.split(","))
    agg = aggregate(args.variant, seeds=seeds)
    if agg is None:
        print(f"No runs found for variant={args.variant} seeds={seeds}")
        return
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"=== {args.variant} across seeds {agg['seeds']} ===")
    print(f"  final test exact : {agg['final_test_mean']:.2f}% ± {agg['final_test_std']:.2f}%")
    print(f"  best  test exact : {agg['best_test_mean']:.2f}% ± {agg['best_test_std']:.2f}%")
    print(f"  per-seed final   : " +
          " ".join(f"s{r['seed']}={r['final_test_exact']:.2f}%" for r in agg["per_run"]))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
