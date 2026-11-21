# collect_trials.py
"""Collate tuner trial logs (<dir>/trial_*/train.log) into one table.

Reads each log directly (no Optuna storage needed): the args dict printed at
start, the parameter count, and every epoch line. Reports best validation
value, test at that epoch, epochs run and status, plus the hyperparameters
that differ between trials.

  python collect_trials.py --dir optuna_ZINC12k
  python collect_trials.py --dir optuna_ZINC12k --top 30 --csv zinc_trials.csv
  python collect_trials.py --dir optuna_ZINC12k --status done,pruned
"""

import argparse
import ast
import csv
import glob
import os
import re
import time

EPOCH_RE = re.compile(
    r"^epoch\s+(\d+)\s.*\|\s*val loss\s+\S+\s+(\S+)\s+([-+0-9.eE]+|nan|inf)\s*\|"
    r"\s*test loss\s+\S+\s+\S+\s+([-+0-9.eE]+|nan|inf)")
PARAMS_RE = re.compile(r"trainable params:\s*([0-9.]+)M")
LOWER_IS_BETTER = {"MAE"}
# Not hyperparameters (differ per trial by construction).
IGNORE = {"device", "save_dir", "checkpoint", "num_workers", "dist_mask_workers"}


def parse_log(path, running_minutes):
    lines = open(path, errors="replace").read().splitlines()
    args = None
    for l in lines[:5]:
        if l.startswith("{") and l.endswith("}"):
            try:
                args = ast.literal_eval(l)
            except (ValueError, SyntaxError):
                pass
            break
    params = next((float(m.group(1)) for m in map(PARAMS_RE.search, lines) if m), None)
    epochs = [m for m in map(EPOCH_RE.search, lines) if m]
    label = epochs[0].group(2) if epochs else None
    vals = [(int(m.group(1)), float(m.group(3)), float(m.group(4))) for m in epochs]

    if any(l.startswith("BEST:") for l in lines):
        status = "done"
    elif any("Traceback" in l or "Error" in l.split(":")[0] for l in lines):
        status = "failed"
    elif time.time() - os.path.getmtime(path) < running_minutes * 60:
        status = "running"
    else:
        status = "pruned"                      # stopped by the tuner (killed)
    err = next((l.strip() for l in reversed(lines)
                if "Error" in l or "error" in l), "") if status == "failed" else ""
    return {"trial": os.path.basename(os.path.dirname(path)), "args": args or {},
            "params_M": params, "label": label, "vals": vals, "status": status,
            "error": err[:80]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Tuner output dir, e.g. optuna_ZINC12k")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--status", default="done,pruned,running,failed",
                   help="Comma list of statuses to show")
    p.add_argument("--direction", choices=["auto", "minimize", "maximize"], default="auto")
    p.add_argument("--running_minutes", type=float, default=15,
                   help="A log without BEST: modified this recently counts as running")
    p.add_argument("--csv", default=None, help="Also write all trials to this CSV")
    a = p.parse_args()

    runs = [parse_log(f, a.running_minutes)
            for f in sorted(glob.glob(os.path.join(a.dir, "trial_*", "train.log")))]
    if not runs:
        print(f"no logs under {a.dir}/trial_*/train.log")
        return
    label = next((r["label"] for r in runs if r["label"]), "metric")
    lower = (label in LOWER_IS_BETTER) if a.direction == "auto" else a.direction == "minimize"
    for r in runs:
        if r["vals"]:
            best = (min if lower else max)(r["vals"], key=lambda v: v[1])
            r.update(best_epoch=best[0], best_val=best[1], test_at_best=best[2],
                     last_epoch=r["vals"][-1][0])
        else:
            r.update(best_epoch=None, best_val=None, test_at_best=None, last_epoch=None)

    # Hyperparameters that vary across trials.
    keys = sorted({k for r in runs for k in r["args"]} - IGNORE)
    varying = [k for k in keys if len({repr(r["args"].get(k)) for r in runs}) > 1
               and k not in ("max_epochs", "patience", "seed")]

    counts = {s: sum(r["status"] == s for r in runs) for s in ("done", "pruned", "running", "failed")}
    print(f"{len(runs)} trials in {a.dir}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))

    show = set(a.status.split(","))
    ranked = sorted([r for r in runs if r["status"] in show and r["best_val"] is not None],
                    key=lambda r: r["best_val"], reverse=not lower)
    print(f"\nTop {min(a.top, len(ranked))} by val {label} "
          f"({'lower' if lower else 'higher'} is better):")
    hdr = f"{'trial':<11} {'status':<8} {'val':>8} {'test':>8} {'best ep':>7} {'last ep':>7} {'params':>7}"
    print(hdr)
    for r in ranked[:a.top]:
        print(f"{r['trial']:<11} {r['status']:<8} {r['best_val']:>8.4f} {r['test_at_best']:>8.4f} "
              f"{r['best_epoch']:>7} {r['last_epoch']:>7} "
              f"{(str(r['params_M']) + 'M') if r['params_M'] else '-':>7}")
        diffs = ", ".join(f"{k}={r['args'].get(k)}" for k in varying if k in r["args"])
        print(f"            {diffs}")

    failed = [r for r in runs if r["status"] == "failed"]
    if failed and "failed" in show:
        print("\nFailed trials:")
        for r in failed:
            print(f"  {r['trial']}: {r['error']}")

    if a.csv:
        cols = ["trial", "status", "best_val", "test_at_best", "best_epoch", "last_epoch",
                "params_M"] + varying
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in sorted(runs, key=lambda r: r["trial"]):
                w.writerow([r.get(c) if c in r else r["args"].get(c) for c in cols])
        print(f"\nWrote {len(runs)} trials to {a.csv}")


if __name__ == "__main__":
    main()

