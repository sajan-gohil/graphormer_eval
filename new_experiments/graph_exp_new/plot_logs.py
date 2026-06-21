#!/usr/bin/env python3
"""
plot_logs.py

Reads all .log files in a given directory (default: current directory),
parses epoch lines, splits on epoch-000 restarts (= new runs), and saves
one PNG per log-file/run pair showing:
  Left panel:  train / val / test Loss
  Right panel: train / val / test AP
"""

import re
import os
import glob
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Regex: matches a line that contains all six metrics
# ---------------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s*\|"
    r".*?train loss\s+([\d.]+)\s+AP\s+([\d.]+)"
    r".*?val loss\s+([\d.]+)\s+AP\s+([\d.]+)"
    r".*?test loss\s+([\d.]+)\s+AP\s+([\d.]+)",
    re.IGNORECASE,
)

COLORS = {"train": "#4C9BE8", "val": "#F4A261", "test": "#57CC99"}
LINE_STYLES = {"train": "-", "val": "--", "test": ":"}
ALPHA_FILL = 0.08

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
def parse_log(path):
    """
    Returns a list of runs.  Each run is:
        {split: {"epochs": [...], "loss": [...], "ap": [...]}}
    A new run starts whenever epoch == 0 appears again.
    """
    runs = []
    current = None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m is None:
                continue

            epoch = int(m.group(1))
            train_loss, train_ap = float(m.group(2)), float(m.group(3))
            val_loss,   val_ap   = float(m.group(4)), float(m.group(5))
            test_loss,  test_ap  = float(m.group(6)), float(m.group(7))

            if epoch == 0 or current is None:
                current = {s: {"epochs": [], "loss": [], "ap": []}
                           for s in ("train", "val", "test")}
                runs.append(current)

            for split, loss, ap in [
                ("train", train_loss, train_ap),
                ("val",   val_loss,   val_ap),
                ("test",  test_loss,  test_ap),
            ]:
                current[split]["epochs"].append(epoch)
                current[split]["loss"].append(loss)
                current[split]["ap"].append(ap)

    return runs


# ---------------------------------------------------------------------------
def plot_run(run, title, out_path):
    fig, (ax_loss, ax_ap) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for split in ("train", "val", "test"):
        data   = run[split]
        epochs = data["epochs"]
        col    = COLORS[split]
        ls     = LINE_STYLES[split]

        ax_loss.plot(epochs, data["loss"],
                     color=col, linestyle=ls, linewidth=1.8,
                     label=split.capitalize(), alpha=0.9)
        ax_loss.fill_between(epochs, data["loss"], alpha=ALPHA_FILL, color=col)

        ax_ap.plot(epochs, data["ap"],
                   color=col, linestyle=ls, linewidth=1.8,
                   label=split.capitalize(), alpha=0.9)
        ax_ap.fill_between(epochs, data["ap"], alpha=ALPHA_FILL, color=col)

    # Mark best-val AP
    val_ap  = run["val"]["ap"]
    val_eps = run["val"]["epochs"]
    if val_ap:
        best_idx = max(range(len(val_ap)), key=lambda i: val_ap[i])
        bx, by = val_eps[best_idx], val_ap[best_idx]
        ax_ap.axvline(bx, color=COLORS["val"], linestyle=":", alpha=0.5, linewidth=1)
        ax_ap.annotate(
            "best val\n{:.4f} @ ep {}".format(by, bx),
            xy=(bx, by),
            xytext=(6, -32),
            textcoords="offset points",
            fontsize=7.5,
            color=COLORS["val"],
            arrowprops=dict(arrowstyle="->", color=COLORS["val"], lw=0.8),
        )

    for ax, ylabel, panel_title in [
        (ax_loss, "Loss",             "Loss  (train / val / test)"),
        (ax_ap,   "Average Precision", "AP    (train / val / test)"),
    ]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(panel_title, fontsize=11, pad=8)
        ax.legend(fontsize=9, framealpha=0.7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> {}".format(out_path))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val/test loss and AP from training log files."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=".",
        help="Directory containing .log files (default: current directory).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for PNGs (default: <log_dir>/plots).",
    )
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    out_dir = args.out_dir or os.path.join(log_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not log_files:
        print("No .log files found in: {}".format(log_dir))
        return

    print("Found {} log file(s).  Output -> {}\n".format(len(log_files), out_dir))

    for log_path in log_files:
        base   = os.path.splitext(os.path.basename(log_path))[0]
        runs   = parse_log(log_path)
        n_runs = len(runs)

        if n_runs == 0:
            print("  [warn] no epoch lines found in: {}".format(log_path))
            continue

        print("  {}: {} run(s)".format(base, n_runs))
        for run_idx, run in enumerate(runs, start=1):
            n_ep     = len(run["train"]["epochs"])
            title    = "{}  -  run {}/{} ({} epochs)".format(
                base, run_idx, n_runs, n_ep)
            out_name = "{}_run{}.png".format(base, run_idx)
            out_path = os.path.join(out_dir, out_name)
            plot_run(run, title, out_path)


if __name__ == "__main__":
    main()
