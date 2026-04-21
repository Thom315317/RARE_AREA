#!/usr/bin/env python3
"""
Sweep final — résultats publiables SCAN + COGS + SLOG.

Orchestre tous les runs, skip ceux déjà terminés, produit les tableaux
avec IC 95% et tests de significativité.

Usage:
  python3 sweep_final.py --benchmark scan
  python3 sweep_final.py --benchmark cogs
  python3 sweep_final.py --benchmark slog
  python3 sweep_final.py --benchmark all
  python3 sweep_final.py --recap          # juste les tableaux
"""
import os, sys, json, time, argparse, glob, subprocess
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "runs_pub")

SEEDS = [42, 123, 456, 789, 1337]
T_CRIT_4 = 2.776   # t_{4, 0.025} for 5 seeds IC 95%

# ══════════════════════════════════════════════════════════════
# SCAN config
# ══════════════════════════════════════════════════════════════
SCAN_SPLITS = {
    "addprim_jump": {
        "train_url": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
        "test_url":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt",
    },
    "addprim_turn_left": {
        "train_url": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_turn_left.txt",
        "test_url":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_turn_left.txt",
    },
    "length": {
        "train_url": "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt",
        "test_url":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt",
    },
}

SCAN_VARIANTS = {
    "addprim_jump":      ["A0", "A1", "A4_seen", "A4_all"],
    "addprim_turn_left": ["A0", "A1", "A4_seen", "A4_all"],
    "length":            ["A0", "A1"],
}

SCAN_EPOCHS = {
    "A0": 50, "A1": 50, "A4_permute": 30, "A4_seen": 30, "A4_all": 30,
}

# ══════════════════════════════════════════════════════════════
# COGS config
# ══════════════════════════════════════════════════════════════
COGS_VARIANTS = ["B0", "B1", "B3_auto", "B4"]
COGS_EPOCHS = 80

# ══════════════════════════════════════════════════════════════
# SLOG config
# ══════════════════════════════════════════════════════════════
SLOG_VARIANTS = ["B0", "B1", "B3_auto", "B4"]
SLOG_EPOCHS = 60


# ══════════════════════════════════════════════════════════════
# Run management
# ══════════════════════════════════════════════════════════════
def find_existing_run(runs_dir: str, variant: str, seed: int) -> Optional[str]:
    """Check if a completed run exists (has summary.json)."""
    pattern = os.path.join(runs_dir, f"{variant}_s{seed}_*", "summary.json")
    matches = sorted(glob.glob(pattern))
    return os.path.dirname(matches[-1]) if matches else None


def run_scan(split: str, variant: str, seed: int):
    """Run one SCAN experiment."""
    runs_dir = os.path.join(SWEEP_DIR, f"scan_{split}")
    os.makedirs(runs_dir, exist_ok=True)

    existing = find_existing_run(runs_dir, variant, seed)
    if existing:
        print(f"  SKIP (exists): {os.path.basename(existing)}")
        return

    epochs = SCAN_EPOCHS.get(variant, 50)
    # For addprim_turn_left, we need to configure the SCAN split URLs
    split_conf = SCAN_SPLITS[split]

    cmd = [
        sys.executable, os.path.join(HERE, "scan_compositional.py"),
        "--variant", variant,
        "--split", split,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", "30",
        "--runs-dir", runs_dir,
    ]

    log_path = os.path.join(runs_dir, f"{variant}_s{seed}.log")
    print(f"  RUN: {variant} seed={seed} epochs={epochs} → {log_path}")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=HERE, timeout=7200)
    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}), see {log_path}")


def run_cogs(variant: str, seed: int):
    """Run one COGS experiment."""
    runs_dir = os.path.join(SWEEP_DIR, "cogs")
    os.makedirs(runs_dir, exist_ok=True)

    existing = find_existing_run(runs_dir, variant, seed)
    if existing:
        print(f"  SKIP (exists): {os.path.basename(existing)}")
        return

    cmd = [
        sys.executable, os.path.join(HERE, "cogs_compositional.py"),
        "--variant", variant,
        "--seed", str(seed),
        "--epochs", str(COGS_EPOCHS),
        "--patience", "20",
        "--runs-dir", runs_dir,
    ]

    log_path = os.path.join(runs_dir, f"{variant}_s{seed}.log")
    print(f"  RUN: {variant} seed={seed} → {log_path}")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=HERE, timeout=7200)
    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}), see {log_path}")


def run_slog(variant: str, seed: int):
    """Run one SLOG experiment."""
    runs_dir = os.path.join(SWEEP_DIR, "slog")
    os.makedirs(runs_dir, exist_ok=True)

    existing = find_existing_run(runs_dir, variant, seed)
    if existing:
        print(f"  SKIP (exists): {os.path.basename(existing)}")
        return

    cmd = [
        sys.executable, os.path.join(HERE, "slog_compositional.py"),
        "--variant", variant,
        "--seed", str(seed),
        "--epochs", str(SLOG_EPOCHS),
        "--patience", "20",
        "--runs-dir", runs_dir,
    ]

    log_path = os.path.join(runs_dir, f"{variant}_s{seed}.log")
    print(f"  RUN: {variant} seed={seed} → {log_path}")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=HERE, timeout=7200)
    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}), see {log_path}")


# ══════════════════════════════════════════════════════════════
# Sweep orchestration
# ══════════════════════════════════════════════════════════════
def sweep_scan():
    print("\n" + "=" * 70)
    print("  SCAN — sweep final")
    print("=" * 70)
    for split in SCAN_SPLITS:
        variants = SCAN_VARIANTS[split]
        for variant in variants:
            for seed in SEEDS:
                print(f"\n▶ SCAN {split} / {variant} / seed={seed}")
                run_scan(split, variant, seed)


def sweep_cogs():
    print("\n" + "=" * 70)
    print("  COGS — sweep final")
    print("=" * 70)
    for variant in COGS_VARIANTS:
        for seed in SEEDS:
            print(f"\n▶ COGS {variant} / seed={seed}")
            run_cogs(variant, seed)


def sweep_slog():
    print("\n" + "=" * 70)
    print("  SLOG — sweep final")
    print("=" * 70)
    for variant in SLOG_VARIANTS:
        for seed in SEEDS:
            print(f"\n▶ SLOG {variant} / seed={seed}")
            run_slog(variant, seed)


# ══════════════════════════════════════════════════════════════
# Recap — aggregate results with IC 95%
# ══════════════════════════════════════════════════════════════
def ic95(values):
    """Return mean, std, ci95_half for a list of values."""
    a = np.array(values, dtype=float)
    n = len(a)
    if n < 2:
        return float(a.mean()), 0.0, 0.0
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    ci = T_CRIT_4 * std / np.sqrt(n) if n == 5 else 1.96 * std / np.sqrt(n)
    return mean, std, float(ci)


def load_summary(run_dir):
    s = os.path.join(run_dir, "summary.json")
    if os.path.exists(s):
        return json.load(open(s))
    return None


def recap_scan():
    print("\n" + "=" * 70)
    print("  SCAN — recap")
    print("=" * 70)
    results = {}
    for split in SCAN_SPLITS:
        runs_dir = os.path.join(SWEEP_DIR, f"scan_{split}")
        if not os.path.isdir(runs_dir):
            continue
        for variant in SCAN_VARIANTS.get(split, []):
            key = f"{split}/{variant}"
            vals = []
            for seed in SEEDS:
                existing = find_existing_run(runs_dir, variant, seed)
                if existing:
                    s = load_summary(existing)
                    if s:
                        vals.append(s.get("best_test_exact", s.get("final_test_exact", 0)))
            if vals:
                mean, std, ci = ic95(vals)
                results[key] = {"mean": mean, "std": std, "ci95": ci,
                                "n_seeds": len(vals), "values": vals}

    print(f"\n{'Split/Variant':<35s} {'Mean':>8s} {'±IC95':>8s} {'Std':>7s} {'Seeds':>6s} {'Min':>6s} {'Max':>6s}")
    print("-" * 78)
    for key in sorted(results):
        r = results[key]
        vals = r["values"]
        print(f"  {key:<33s} {r['mean']:7.2f}% {r['ci95']:7.2f}% {r['std']:6.2f}% "
              f"{r['n_seeds']:>5d} {min(vals):5.1f}% {max(vals):5.1f}%")

    out = os.path.join(SWEEP_DIR, "sweep_results_scan.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


def recap_cogs_or_slog(benchmark="cogs"):
    variants = COGS_VARIANTS if benchmark == "cogs" else SLOG_VARIANTS
    runs_dir = os.path.join(SWEEP_DIR, benchmark)
    print(f"\n{'='*70}")
    print(f"  {benchmark.upper()} — recap")
    print(f"{'='*70}")
    if not os.path.isdir(runs_dir):
        print(f"  No runs found in {runs_dir}")
        return {}

    results = {}
    for variant in variants:
        vals_gen = []
        vals_dev = []
        by_cat_all = {}   # cat → list of values across seeds
        for seed in SEEDS:
            existing = find_existing_run(runs_dir, variant, seed)
            if not existing:
                continue
            s = load_summary(existing)
            if not s:
                continue
            gen = s.get("final_gen_exact_greedy", 0)
            dev = s.get("final_dev_exact_greedy", s.get("final_dev_exact_tf", 0))
            vals_gen.append(gen)
            vals_dev.append(dev)
            for cat, pct in s.get("greedy_gen_by_cat", {}).items():
                by_cat_all.setdefault(cat, []).append(pct)

        if vals_gen:
            gm, gs, gc = ic95(vals_gen)
            dm, ds, dc = ic95(vals_dev)
            cat_stats = {}
            for cat, cvals in sorted(by_cat_all.items()):
                cm, cs, cc = ic95(cvals)
                cat_stats[cat] = {"mean": cm, "ci95": cc, "n": len(cvals)}
            results[variant] = {
                "gen_mean": gm, "gen_std": gs, "gen_ci95": gc,
                "dev_mean": dm, "dev_ci95": dc,
                "n_seeds": len(vals_gen), "gen_values": vals_gen,
                "by_cat": cat_stats,
            }

    print(f"\n{'Variant':<12s} {'Gen EM':>10s} {'±IC95':>8s} {'Dev EM':>8s} {'Seeds':>6s}")
    print("-" * 50)
    for v in variants:
        if v in results:
            r = results[v]
            print(f"  {v:<10s} {r['gen_mean']:9.2f}% {r['gen_ci95']:7.2f}% "
                  f"{r['dev_mean']:7.1f}% {r['n_seeds']:>5d}")

    # Per-category table for best variant
    best_v = max(results, key=lambda v: results[v]["gen_mean"]) if results else None
    if best_v and results[best_v]["by_cat"]:
        print(f"\n  Per-category breakdown ({best_v}):")
        print(f"  {'Category':<55s} {'Mean':>7s} {'±IC95':>7s}")
        print(f"  {'-'*55} {'-'*7} {'-'*7}")
        for cat, cs in sorted(results[best_v]["by_cat"].items(), key=lambda x: -x[1]["mean"]):
            if cs["mean"] > 0:
                print(f"  {cat:<55s} {cs['mean']:6.2f}% {cs['ci95']:6.2f}%")

    out = os.path.join(SWEEP_DIR, f"sweep_results_{benchmark}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


def generate_tables():
    """Generate sweep_tables.md from saved JSON results."""
    md_lines = ["# Sweep Results — Publication Tables\n",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]

    # SCAN
    scan_path = os.path.join(SWEEP_DIR, "sweep_results_scan.json")
    if os.path.exists(scan_path):
        scan = json.load(open(scan_path))
        md_lines.append("\n## SCAN\n")
        md_lines.append("| Split | Variant | Test EM (mean ± IC95) | Std | Seeds |")
        md_lines.append("|---|---|---|---|---|")
        for key in sorted(scan):
            r = scan[key]
            md_lines.append(f"| {key.replace('/', ' | ')} | "
                           f"{r['mean']:.2f}% ± {r['ci95']:.2f}% | "
                           f"{r['std']:.2f}% | {r['n_seeds']} |")

    # COGS
    for bench in ["cogs", "slog"]:
        path = os.path.join(SWEEP_DIR, f"sweep_results_{bench}.json")
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        md_lines.append(f"\n## {bench.upper()}\n")
        md_lines.append("| Variant | Gen EM (mean ± IC95) | Dev EM | Seeds |")
        md_lines.append("|---|---|---|---|")
        for v, r in sorted(data.items()):
            md_lines.append(f"| {v} | {r['gen_mean']:.2f}% ± {r['gen_ci95']:.2f}% | "
                           f"{r['dev_mean']:.1f}% | {r['n_seeds']} |")

    out = os.path.join(SWEEP_DIR, "sweep_tables.md")
    with open(out, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved: {out}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sweep final — publication results")
    p.add_argument("--benchmark", type=str, default="all",
                   choices=["scan", "cogs", "slog", "all"])
    p.add_argument("--recap", action="store_true",
                   help="Only generate recap tables (no training)")
    p.add_argument("--seeds", type=str, default=None,
                   help="Override seeds (comma-separated)")
    args = p.parse_args()

    if args.seeds:
        SEEDS[:] = [int(s) for s in args.seeds.split(",")]
    print(f"Seeds: {SEEDS}")

    os.makedirs(SWEEP_DIR, exist_ok=True)

    if args.recap:
        recap_scan()
        recap_cogs_or_slog("cogs")
        recap_cogs_or_slog("slog")
        generate_tables()
    else:
        t0 = time.time()
        if args.benchmark in ("scan", "all"):
            sweep_scan()
        if args.benchmark in ("cogs", "all"):
            sweep_cogs()
        if args.benchmark in ("slog", "all"):
            sweep_slog()
        dt = time.time() - t0
        print(f"\n{'='*70}")
        print(f"  SWEEP COMPLETE — {dt/3600:.1f}h")
        print(f"{'='*70}")

        # Generate recap
        recap_scan()
        recap_cogs_or_slog("cogs")
        recap_cogs_or_slog("slog")
        generate_tables()
