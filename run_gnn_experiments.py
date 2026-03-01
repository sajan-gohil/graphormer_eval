"""
Run the Graphormer experiment with five different GNN structural settings to
study how each approach affects long-range dependency handling.

Settings
--------
none              – baseline Graphormer with no structural modification
random_edges      – add random shortcut edges (≈10 % of existing edges)
                    to create graph-wide shortcuts that reduce effective
                    pairwise distances
bounded_diameter  – deterministically add edges so the diameter (longest
                    shortest path) is at most 5 hops
relay_nodes       – insert virtual relay nodes between distant node pairs,
                    halving their effective communication distance
spectral_bias     – enrich the Graphormer attention bias with a Laplacian
                    spectral similarity term derived from the graph's
                    eigenvectors (Laplacian Positional Encodings)

Usage
-----
# Run all transforms on Cora
python run_gnn_experiments.py --dataset_name cora

# Run a specific subset
python run_gnn_experiments.py --dataset_name cora --transforms none random_edges spectral_bias

# Dry-run: just print the commands that would be executed
python run_gnn_experiments.py --dataset_name cora --dry_run
"""

import argparse
import subprocess
import sys

TRANSFORMS = [
    "none",
    "random_edges",
    "bounded_diameter",
    "relay_nodes",
    "spectral_bias",
]

TRANSFORM_DESCRIPTIONS = {
    "none":             "Baseline Graphormer (no structural modification)",
    "random_edges":     "Random shortcut edges (~10% of existing edges added)",
    "bounded_diameter": "Deterministic edges to enforce diameter < 6",
    "relay_nodes":      "Virtual relay nodes inserted between distant pairs",
    "spectral_bias":    "Laplacian spectral embeddings added to attention bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Graphormer experiments with different GNN structural settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cora",
        help="Dataset to experiment on (default: cora)",
    )
    parser.add_argument(
        "--transforms", nargs="+", default=TRANSFORMS, choices=TRANSFORMS,
        help="Transforms to run (default: all five)",
    )
    parser.add_argument(
        "--experiment_dir", type=str, default="./gnn_experiments",
        help="Base directory for experiment outputs (default: ./gnn_experiments)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="Batch size passed to train_graphormer.py (default: 512)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate passed to train_graphormer.py (default: 2e-5)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data-loading workers (default: 0)",
    )
    parser.add_argument(
        "--onscreen_logs", action="store_true",
        help="Print logs to screen instead of log files",
    )
    parser.add_argument(
        "--train_script", type=str, default="train_graphormer.py",
        help="Path to the training script (default: train_graphormer.py)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def build_command(transform: str, args: argparse.Namespace) -> list[str]:
    """Build the command list for a single training run."""
    exp_name = f"{args.dataset_name}_{transform}"
    cmd = [
        sys.executable, args.train_script,
        "--dataset_name", args.dataset_name,
        "--gnn_transform", transform,
        "--name", exp_name,
        "--experiment_dir", args.experiment_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_workers", str(args.num_workers),
    ]
    if args.onscreen_logs:
        cmd.append("--onscreen_logs")
    return cmd


def run_experiment(transform: str, args: argparse.Namespace) -> int:
    """Launch a single training run and return the exit code."""
    cmd = build_command(transform, args)
    desc = TRANSFORM_DESCRIPTIONS.get(transform, transform)

    print(f"\n{'=' * 65}")
    print(f"  Transform : {transform}")
    print(f"  Description: {desc}")
    print(f"  Command   : {' '.join(cmd)}")
    print(f"{'=' * 65}\n")

    if args.dry_run:
        print("[dry-run] Skipping execution.")
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"[WARNING] Experiment '{transform}' exited with code "
            f"{result.returncode}"
        )
    return result.returncode


def main() -> None:
    args = parse_args()

    print(f"\nGNN Structural Experiment Runner")
    print(f"  Dataset   : {args.dataset_name}")
    print(f"  Transforms: {', '.join(args.transforms)}")
    print(f"  Output dir: {args.experiment_dir}")
    if args.dry_run:
        print("  [dry-run mode – no training will be launched]")

    results: dict[str, str] = {}
    for transform in args.transforms:
        rc = run_experiment(transform, args)
        results[transform] = "OK" if rc == 0 else f"FAILED (exit {rc})"

    # Print summary table
    print(f"\n{'=' * 65}")
    print("Experiment Summary")
    print(f"{'=' * 65}")
    print(f"  {'Transform':<22}  {'Description':<38}  Status")
    print(f"  {'-'*22}  {'-'*38}  ------")
    for transform, status in results.items():
        desc = TRANSFORM_DESCRIPTIONS.get(transform, "")
        print(f"  {transform:<22}  {desc:<38}  {status}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
