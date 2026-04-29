#!/usr/bin/env python3
"""Petit utilitaire — écrit baseline_sanity.json à partir d'un metrics.json
SLOG (post-train).

Usage :
  python3 write_baseline_sanity.py \\
      --piste p1_lambda005 \\
      --run-dir runs/B4_slog_jepa_p1_lambda005/B4_s42_*
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piste", required=True,
                    help="Identifiant de la piste (ex: p1_lambda005)")
    ap.add_argument("--run-dir", required=True,
                    help="Dossier du run (peut contenir un glob *)")
    args = ap.parse_args()

    matches = glob.glob(args.run_dir)
    if not matches:
        print(f"ERROR: no run dir matching {args.run_dir}")
        raise SystemExit(1)
    run_dir = matches[0]

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"ERROR: no metrics.json at {metrics_path}")
        raise SystemExit(1)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # metrics is a list of per-epoch dicts ; final entry has the final values.
    last = metrics[-1] if isinstance(metrics, list) else metrics
    first = metrics[0] if isinstance(metrics, list) else metrics

    # Extract final greedy from summary in the same dir if available
    greedy = {}
    summ_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summ_path):
        with open(summ_path, "r") as f:
            summ = json.load(f)
        greedy["final_dev_greedy"] = summ.get("final_dev_exact_greedy")
        greedy["final_gen_greedy"] = summ.get("final_gen_exact_greedy")
        greedy["best_gen_tf"] = summ.get("best_gen_exact_tf")
        greedy["best_dev_exact"] = summ.get("best_dev_exact")

    out = {
        "piste": args.piste,
        "run_dir": run_dir,
        "best_gen_tf": greedy.get("best_gen_tf"),
        "gen_greedy": greedy.get("final_gen_greedy"),
        "dev_greedy": greedy.get("final_dev_greedy"),
        "best_dev": greedy.get("best_dev_exact"),
        "jepa_loss_first_epoch": first.get("jepa_loss_train_avg"),
        "jepa_loss_final": last.get("jepa_loss_train_avg"),
        "epochs_run": last.get("epoch", 0) + 1,
    }
    # gen_greedy est en pourcentage (ex: 26.86), pas en fraction.
    out["task_acc_within_range"] = (
        out["gen_greedy"] is not None and 27.0 <= out["gen_greedy"] <= 33.0
    )

    out_path = os.path.join(run_dir, "baseline_sanity.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
