"""
Bayesian hyperparameter search using Optuna (TPE sampler) for the
HopMaskedTransformerModel.

Designed to run *fast* by:
  - Training only a fraction of the train set (--train_fraction, default 0.25).
  - Running fewer epochs (--max_epochs, default 30).
  - Reducing patience for early stopping (--patience, default 10).
  - Using a fast, median-based pruner to kill unpromising trials early.

The objective is the best validation metric observed during the short run.
For Peptides-func (AP — higher is better) it maximises; for
Peptides-struct (MAE — lower is better) it minimises.

Usage:
    python optuna_hop_masked_search.py --dataset Peptides-func --n_trials 50

    # Use more data or longer training for a finer search
    python optuna_hop_masked_search.py --dataset Peptides-func --n_trials 100 \\
        --train_fraction 0.5 --max_epochs 50

    # Resume a study stored in an SQLite DB
    python optuna_hop_masked_search.py --dataset Peptides-func --n_trials 50 \\
        --study_name my_study --storage sqlite:///optuna_hop_masked.db

The best hyperparameters are printed at the end and also saved to
``optuna_hop_masked_best.json`` (or the path given by --output_json).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch
torch.set_float32_matmul_precision("high")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from data import get_loaders
from metrics import build_task
from model_hop_masked_transformer import HopMaskedTransformerModel
from optim_utils import build_grouped_optimizer_and_scheduler


# ===================================================================
# Argument parser — meta-level controls only (search range stays here)
# ===================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Optuna Bayesian hyper-parameter search for "
                    "HopMaskedTransformerModel")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=["Peptides-func", "Peptides-struct", "PascalVOC-SP"])
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # Speed-up knobs
    p.add_argument("--train_fraction", type=float, default=0.25,
                   help="Fraction of train set to use per trial (0 < frac <= 1).")
    p.add_argument("--max_epochs", type=int, default=30,
                   help="Max epochs per trial (kept small for speed).")
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience per trial.")
    p.add_argument("--batch_size", type=int, default=32)

    # Optuna
    p.add_argument("--n_trials", type=int, default=50,
                   help="Target total number of Optuna trials.")
    p.add_argument("--study_name", type=str,
                   default="hop_masked_hparam_search")
    p.add_argument("--storage", type=str,
                   default="sqlite:///optuna_hop_masked.db",
                   help="Optuna storage URL (default: SQLite for persistence "
                        "across restarts).")
    p.add_argument("--output_json", type=str,
                   default="optuna_hop_masked_best.json",
                   help="Where to dump the best params.")
    p.add_argument("--pruning", action="store_true", default=True,
                   help="Enable Optuna median pruner (default: on).")
    p.add_argument("--no_pruning", action="store_false", dest="pruning")

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)
    return p


# ===================================================================
# Search space definition
# ===================================================================

def suggest_hparams(trial: optuna.Trial) -> dict:
    """Define the Bayesian search space for the hop-masked transformer.

    Returns a dict with all model / training hyper-parameters.
    The ranges are centred around the user's baseline config:
        --max_hops 40 --num_heads 40 --num_layers 1
        --hop_mode window --hop_window 1 --hidden_dim 640
    """

    hp = {}

    # ---- Architecture core -------------------------------------------
    # num_heads must divide hidden_dim.  We suggest both and constrain
    # below; invalid combos (head/dim mismatch) are caught and pruned.
    hp["hidden_dim"] = trial.suggest_categorical(
        "hidden_dim", [128, 256, 320, 384, 512, 640])
    hp["num_heads"] = trial.suggest_categorical(
        "num_heads", [4, 8, 16, 20, 32, 40])
    hp["ffn_ratio"] = trial.suggest_categorical("ffn_ratio", [2, 4])
    hp["num_layers"] = trial.suggest_int("num_layers", 1, 6)
    hp["dropout"] = trial.suggest_float("dropout", 0.05, 0.4, step=0.05)

    # ---- Hop-to-head assignment --------------------------------------
    hp["hop_mode"] = trial.suggest_categorical(
        "hop_mode", ["contiguous", "window"])
    hp["hop_window"] = trial.suggest_int("hop_window", 1, 5)
    hp["num_global_heads"] = trial.suggest_int("num_global_heads", 0, 4)

    # ---- Pooling / task head -----------------------------------------
    hp["graph_pool"] = trial.suggest_categorical(
        "graph_pool", ["sum", "mean"])

    # ---- Training ----------------------------------------------------
    hp["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    hp["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-5, 1e-2, log=True)

    return hp


# ===================================================================
# Train / eval helpers (adapted from train_hop_masked_transformer.py)
# ===================================================================

def _move_batch(batch, device):
    pyg_batch, dist_masks, node_masks = batch
    return pyg_batch.to(device), dist_masks.to(device), node_masks.to(device)


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None,
              grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds_acc, labels_acc = [], [], []
    for batch in loader:
        pyg_batch, dist_masks, node_masks = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, _ = model(pyg_batch, dist_masks, node_masks)
            loss = task.loss(logits, pyg_batch.y)

        if is_train:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(pyg_batch.y))

    y_pred = np.concatenate(preds_acc, axis=0)
    y_true = np.concatenate(labels_acc, axis=0)
    metric = task.compute_metric(y_pred, y_true)
    return float(np.mean(losses)), float(metric)


# ===================================================================
# Objective
# ===================================================================

MIN_BATCH_SIZE = 4   # floor for OOM retry


def make_objective(args, train_loader, val_loader, task):
    """Return a closure that Optuna calls for each trial."""

    def _build_model(hp, task, args):
        """Construct the model; raises on invalid combos."""
        return HopMaskedTransformerModel(
            hidden_dim=hp["hidden_dim"],
            num_heads=hp["num_heads"],
            ffn_ratio=hp["ffn_ratio"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            max_hops=args.max_hops,
            hop_mode=hp["hop_mode"],
            hop_window=hp["hop_window"],
            num_global_heads=hp["num_global_heads"],
            output_dim=task.output_dim,
            graph_pool=hp["graph_pool"],
            task_level=task.level,
            dataset_name=args.dataset,
            lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        ).to(args.device)

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hparams(trial)

        # ---- Build model ----------------------------------------
        try:
            model = _build_model(hp, task, args)
        except Exception as e:
            # Invalid combos (e.g. head/dim mismatch) → prune trial
            print(f"  [trial {trial.number}] Model build failed: {e}")
            print(f"  [trial {trial.number}] params: {hp}")
            raise optuna.TrialPruned()

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        trial.set_user_attr("n_params", n_params)

        # Start with the original batch size; halve on OOM.
        cur_batch_size = args.batch_size
        cur_train_loader = train_loader
        cur_val_loader = val_loader

        while True:   # OOM-retry loop
            total_steps = max(args.max_epochs * len(cur_train_loader), 1)
            optimizer, scheduler = build_grouped_optimizer_and_scheduler(
                named_parameters=list(model.named_parameters()),
                lr_max=hp["lr"],
                lr_min=1e-6,
                weight_decay=hp["weight_decay"],
                total_steps=total_steps,
                warmup_ratio=0.05,
            )

            best_val = -float("inf") if task.higher_is_better else float("inf")
            epochs_since_improve = 0
            oom_hit = False

            for epoch in range(args.max_epochs):
                t0 = time.time()

                try:
                    tr_loss, tr_metric = run_epoch(
                        model, cur_train_loader, task, args.device,
                        optimizer=optimizer, scheduler=scheduler,
                        grad_clip=1.0,
                    )
                    va_loss, va_metric = run_epoch(
                        model, cur_val_loader, task, args.device)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                        oom_hit = True
                        break
                    raise

                dt = time.time() - t0

                improved = (va_metric > best_val if task.higher_is_better
                            else va_metric < best_val)
                if improved:
                    best_val = va_metric
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

                # Report intermediate value to the pruner
                trial.report(va_metric, epoch)

                ml = task.metric_label
                print(
                    f"  [trial {trial.number}] epoch {epoch:02d} {dt:4.1f}s "
                    f"(bs={cur_batch_size}) | "
                    f"train {ml} {tr_metric:.4f} loss {tr_loss:.4f} | "
                    f"val {ml} {va_metric:.4f} | best {best_val:.4f}",
                    flush=True)

                if trial.should_prune():
                    print(f"  [trial {trial.number}] PRUNED at epoch {epoch}")
                    raise optuna.TrialPruned()

                if epochs_since_improve >= args.patience:
                    print(f"  [trial {trial.number}] early stop at epoch {epoch}")
                    break

            if not oom_hit:
                # Training finished (or early-stopped) — done.
                break

            # ---- OOM retry: halve batch size -------------------------
            new_bs = cur_batch_size // 2
            if new_bs < MIN_BATCH_SIZE:
                print(f"  [trial {trial.number}] OOM at bs={cur_batch_size}, "
                      f"already at minimum ({MIN_BATCH_SIZE}) — pruning.")
                print(f"  [trial {trial.number}] params: {hp}")
                del model, optimizer, scheduler
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            print(f"  [trial {trial.number}] OOM at bs={cur_batch_size} → "
                  f"retrying with bs={new_bs}")
            del optimizer, scheduler
            torch.cuda.empty_cache()

            # Rebuild loaders with the smaller batch size
            cur_batch_size = new_bs
            cur_train_loader = _rebuild_loader(train_loader, cur_batch_size)
            cur_val_loader = _rebuild_loader(val_loader, cur_batch_size)

            # Re-init model weights so it starts fresh
            del model
            torch.cuda.empty_cache()
            model = _build_model(hp, task, args)

        trial.set_user_attr("actual_batch_size", cur_batch_size)

        # Clean up GPU memory between trials
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

        return best_val

    return objective


# ===================================================================
# Data subsetting / rebuilding helpers
# ===================================================================

def _subset_loader(loader, fraction, seed=0):
    """Return a new DataLoader that uses only ``fraction`` of the dataset."""
    from torch.utils.data import Subset, DataLoader as TorchDataLoader
    from functools import partial
    from data import collate_with_dist_masks

    ds = loader.dataset
    n = len(ds)
    n_sub = max(1, int(n * fraction))

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)[:n_sub]

    subset = Subset(ds, indices)
    collate_fn = partial(collate_with_dist_masks,
                         max_hops=loader.collate_fn.keywords.get("max_hops", 40)
                         if hasattr(loader.collate_fn, "keywords") else 40)
    return TorchDataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        collate_fn=collate_fn,
    )


def _rebuild_loader(loader, new_batch_size):
    """Return a copy of *loader* with a different batch_size."""
    from torch.utils.data import DataLoader as TorchDataLoader
    return TorchDataLoader(
        loader.dataset,
        batch_size=new_batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
    )


# ===================================================================
# Main
# ===================================================================

def main():
    args = build_parser().parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task = build_task(args.dataset)
    print(f"[{args.dataset}] task={task.task_type}  metric={task.metric_name} "
          f"(higher_is_better={task.higher_is_better})")

    # ---- Load data (full) then subsample train -----------------------
    train_loader, val_loader, _, _, _, _ = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=True,
        max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe,
        lap_pe_dim=args.lap_pe_dim,
        dataset_name=args.dataset,
    )

    if args.train_fraction < 1.0:
        orig_len = len(train_loader.dataset)
        train_loader = _subset_loader(
            train_loader, args.train_fraction, seed=args.seed)
        print(f"Subsampled train: {orig_len} → {len(train_loader.dataset)} "
              f"({args.train_fraction:.0%})")

    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"Max epochs/trial: {args.max_epochs}  Patience: {args.patience}")
    print(f"Optuna trials: {args.n_trials}  Pruning: {args.pruning}")
    print("=" * 72)

    # ---- Create Optuna study -----------------------------------------
    direction = "maximize" if task.higher_is_better else "minimize"
    pruner = (MedianPruner(
        n_startup_trials=5,      # don't prune the first 5 trials
        n_warmup_steps=5,        # don't prune any trial before epoch 5
        interval_steps=1,
    ) if args.pruning else optuna.pruners.NopPruner())

    # First create/load the study with a temporary sampler to inspect
    # existing trials, then recreate the sampler with an offset seed so
    # the TPE quasi-random startup phase doesn't replay old configs.
    study = optuna.create_study(
        study_name=args.study_name,
        direction=direction,
        sampler=TPESampler(seed=args.seed),
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        # Offset the seed so the sampler's initial random exploration
        # produces fresh configs instead of repeating old ones.
        study.sampler = TPESampler(seed=args.seed + n_existing)
        print(f"Resuming study '{args.study_name}' — "
              f"{n_existing} existing trials found.")

    n_remaining = max(0, args.n_trials - n_existing)
    if n_remaining == 0:
        print(f"Already have {n_existing} >= {args.n_trials} trials. "
              f"Nothing to do (increase --n_trials to run more).")
    else:
        print(f"Will run {n_remaining} new trials "
              f"(target total: {args.n_trials}, existing: {n_existing}).")

    # ---- Run search --------------------------------------------------
    objective = make_objective(args, train_loader, val_loader, task)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)

    # ---- Report ------------------------------------------------------
    print("\n" + "=" * 72)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 72)

    best = study.best_trial
    print(f"\nBest trial #{best.number}")
    print(f"  Value ({task.metric_label}): {best.value:.6f}")
    print(f"  Params ({len(best.params)}):")
    for k, v in sorted(best.params.items()):
        print(f"    {k}: {v}")
    if "n_params" in best.user_attrs:
        print(f"  Model size: {best.user_attrs['n_params']/1e6:.3f}M params")
    if "actual_batch_size" in best.user_attrs:
        print(f"  Actual batch size: {best.user_attrs['actual_batch_size']}")

    # Save best params
    result = {
        "dataset": args.dataset,
        "metric": task.metric_name,
        "direction": direction,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED]),
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {args.output_json}")

    # ---- Print top-5 trials ------------------------------------------
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if task.higher_is_better:
        completed.sort(key=lambda t: t.value, reverse=True)
    else:
        completed.sort(key=lambda t: t.value)

    print(f"\nTop {min(5, len(completed))} trials:")
    for i, t in enumerate(completed[:5]):
        print(f"  #{t.number}  {task.metric_label}={t.value:.4f}  "
              f"params: { {k: v for k, v in sorted(t.params.items())} }")

    # ---- Print CLI command for best config ---------------------------
    bp = best.params
    cmd_parts = ["python train_hop_masked_transformer.py"]
    cmd_parts.append(f"--dataset {args.dataset}")
    cmd_parts.append(f"--max_hops {args.max_hops}")
    if args.use_lap_pe:
        cmd_parts.append("--use_lap_pe")
        cmd_parts.append(f"--lap_pe_dim {args.lap_pe_dim}")

    # Map search params to CLI flags
    direct_flags = [
        "hidden_dim", "num_heads", "ffn_ratio", "num_layers", "dropout",
        "hop_mode", "hop_window", "num_global_heads", "graph_pool",
        "lr", "weight_decay",
    ]
    for flag in direct_flags:
        if flag in bp:
            cmd_parts.append(f"--{flag} {bp[flag]}")

    cmd_parts.append("--batch_size 64")
    cmd_parts.append("--max_epochs 200")
    cmd_parts.append("--patience 40")
    cmd_parts.append("--save_dir checkpoints_hop_masked_optuna_best")

    print(f"\nRecommended full training command:\n")
    print("  " + " \\\n    ".join(cmd_parts))
    print()


if __name__ == "__main__":
    main()
