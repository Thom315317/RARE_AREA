#!/usr/bin/env python3
"""Sélectionne le meilleur baseline SLOG parmi les 3 pistes par gen_greedy.

Usage :
  python3 select_best_slog_baseline.py \\
      --pistes p1_lambda005,p2_epochs90,p3_ema \\
      --runs-prefix runs/B4_slog_jepa_ \\
      --output runs/best_slog_baseline.json
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pistes", default="p1_lambda005,p2_epochs90,p3_ema")
    ap.add_argument("--runs-prefix", default="runs/B4_slog_jepa_")
    ap.add_argument("--output", default="runs/best_slog_baseline.json")
    args = ap.parse_args()

    pistes = [p.strip() for p in args.pistes.split(",") if p.strip()]
    baselines = {}
    for piste in pistes:
        sanity_files = glob.glob(os.path.join(
            args.runs_prefix + piste, "*", "baseline_sanity.json"))
        if not sanity_files:
            print(f"  [{piste}] no baseline_sanity.json found")
            continue
        with open(sanity_files[0], "r") as f:
            d = json.load(f)
        baselines[piste] = d
        gg = d.get("gen_greedy")
        gg_str = f"{100*gg:.2f}%" if gg is not None else "N/A"
        bg = d.get("best_gen_tf")
        bg_str = f"{100*bg:.2f}%" if bg is not None else "N/A"
        print(f"  [{piste}] gen_greedy={gg_str}  best_gen_tf={bg_str}  "
              f"jepa_final={d.get('jepa_loss_final')}")

    if not baselines:
        print("ERROR: no baseline found")
        raise SystemExit(1)

    # Pick best by gen_greedy (None → -inf for safety)
    def _key(item):
        v = item[1].get("gen_greedy")
        return v if v is not None else -1.0
    best_piste, best = max(baselines.items(), key=_key)
    print()
    print(f"Best baseline: {best_piste} "
          f"(gen_greedy={100*best['gen_greedy']:.2f}%)")
    print(f"  run_dir = {best['run_dir']}")
    print(f"  task_acc_within_range = {best['task_acc_within_range']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"choice": best_piste,
                   "metrics": best,
                   "all_baselines": baselines}, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
