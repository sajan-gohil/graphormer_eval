# run_synthetic.py
"""Train/test the light hop-masked model on synthetic_suite datasets.

Reuses model_hop_masked_light, data.py (distance/predecessor caches, collate)
and train_hop_masked_light (CLI args, run_epoch, diagnostics) without changing
them. All model flags of train_hop_masked_light.py are accepted.

Example:
  python run_synthetic.py --data synthetic/t3_tree_n200_d2_k10_e8_noise0.0_decauto \\
      --seeds 0,1,2 --max_hops 32 --use_path_embeddings --use_edge_features \\
      --results synthetic_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

import data as data_mod
from data import (DistMaskDataset, collate_with_dist_masks,
                  precompute_distance_matrices, precompute_predecessors)
from metrics import build_task
from model_hop_masked_light import HopMaskedTransformerModel
from optim_utils import build_grouped_optimizer_and_scheduler
from synthetic_suite import SyntheticPathDataset
from train_hop_masked_light import attn_diagnostics, build_parser, run_epoch

SYN_NAME = "Synthetic-Path"


def register(meta: dict) -> dict:
    """Runtime registry entry (node encoders read data.GRAPH_DATASETS)."""
    info = {"output_dim": 2, "task_type": "multiclass", "level": "node",
            "node_encoder": "linear", "node_feat_dim": meta["node_feat_dim"],
            "edge_feat_dim": meta["edge_feat_dim"], "metric_name": "accuracy",
            "source": "synthetic"}
    data_mod.GRAPH_DATASETS[SYN_NAME] = info
    return dict(info, name=SYN_NAME)


def build_loaders(root, args):
    splits = {sp: SyntheticPathDataset(root, sp) for sp in ("train", "val", "test")}
    cache = os.path.join(root, f"dist_cache_h{args.max_hops}")
    loaders = {}
    for sp, ds in splits.items():
        dm = precompute_distance_matrices(ds, cache, sp, args.max_hops,
                                          args.dist_mask_workers)
        pr = (precompute_predecessors(ds, dm, cache, sp, args.dist_mask_workers)
              if args.use_path_embeddings else None)
        collate = partial(collate_with_dist_masks, max_hops=args.max_hops,
                          use_path_embeddings=args.use_path_embeddings,
                          dist_as_index=True)
        loaders[sp] = TorchDataLoader(DistMaskDataset(ds, dm, pr),
                                      batch_size=args.batch_size, shuffle=(sp == "train"),
                                      num_workers=args.num_workers, collate_fn=collate)
    return loaders, splits["train"].meta


def build_model(args, info):
    return HopMaskedTransformerModel(
        hidden_dim=args.hidden_dim, num_heads=args.num_heads, ffn_ratio=args.ffn_ratio,
        num_layers=args.num_layers, dropout=args.dropout, max_hops=args.max_hops,
        hop_mode=args.hop_mode, hop_window=args.hop_window, hop_file=args.hop_file,
        num_global_heads=args.num_global_heads, output_dim=info["output_dim"],
        graph_pool=args.graph_pool, task_level="node", dataset_name=SYN_NAME,
        node_feat_dim=info["node_feat_dim"], block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop, norm_type=args.norm_type,
        mask_type=args.mask_type, adj_self_loops=args.adj_self_loops,
        num_post_gat_layers=args.num_post_gat_layers, num_gat_heads=args.num_gat_heads,
        use_edge_features=args.use_edge_features, edge_feat_dim=info["edge_feat_dim"],
        blend_adj_power=args.blend_adj_power, use_edge_bias=args.use_edge_bias,
        multihop_attn=args.multihop_attn, multihop_readout=args.multihop_readout,
        multihop_include_global=not args.multihop_no_global,
        use_path_embeddings=args.use_path_embeddings,
    ).to(args.device)


def train_one_seed(args, info, loaders, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    task = build_task(SYN_NAME, dataset_info=info)
    task.loss_fn = task.loss_fn.to(args.device)
    model = build_model(args, info)
    total_steps = max(args.max_epochs * len(loaders["train"]), 1)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=list(model.named_parameters()), lr_max=args.lr,
        lr_min=args.lr_min, weight_decay=args.weight_decay,
        total_steps=total_steps, warmup_ratio=args.warmup_ratio)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.reduce_lr_patience,
        min_lr=args.lr_min)

    best = {"val": -1.0, "test": None, "train": None, "epoch": -1}
    best_state, since = None, 0
    for epoch in range(args.max_epochs):
        t0 = time.time()
        _, tr = run_epoch(model, loaders["train"], task, args.device, optimizer=optimizer,
                          scheduler=scheduler, grad_clip=args.grad_clip, amp=args.amp)
        _, va = run_epoch(model, loaders["val"], task, args.device, amp=args.amp)
        _, te = run_epoch(model, loaders["test"], task, args.device, amp=args.amp)
        plateau.step(va)
        if va > best["val"]:
            best = {"val": va, "test": te, "train": tr, "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        print(f"seed {seed} epoch {epoch:03d} | {time.time() - t0:5.1f}s | "
              f"train acc {tr:.4f} | val acc {va:.4f} | test acc {te:.4f} | "
              f"best val {best['val']:.4f} (ep {best['epoch']})", flush=True)
        if since >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        # Path-gate value per layer (0 at init): shows whether P is used at all.
        best["path_gates"] = [round(float(l.attn.path_gate_scale.item()), 5)
                              for l in model.layers
                              if hasattr(l.attn, "path_gate_scale")]
        if best["path_gates"]:
            print(f"seed {seed} path gates (best ckpt): {best['path_gates']}", flush=True)
        if args.diagnostics:
            attn_diagnostics(model, loaders["val"], args.device, tag=f"seed {seed} best")
    return best


def main():
    p = build_parser()                       # all train_hop_masked_light flags
    p.add_argument("--data", required=True, help="Dataset dir from synthetic_suite.py")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--results", type=str, default=None,
                   help="Append one JSON line per run (config + mean/std)")
    p.add_argument("--diagnostics", action="store_true", default=False)
    p.add_argument("--auto_max_hops", action="store_true", default=False,
                   help="max_hops = clamp(ceil(diameter) + 1, max_dist + 1, "
                        "--max_hops_cap) from the dataset's meta.json")
    p.add_argument("--max_hops_cap", type=int, default=48)
    p.set_defaults(dataset="Peptides-func")  # unused; dataset comes from --data
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    # Largest source-target distance in the dataset (older metas: k).
    max_dist = int(meta.get("max_dist", 1 if meta["task"] == 2 else meta["k"]))
    if args.auto_max_hops:
        diam = meta.get("stats", {}).get("diameter") or max_dist
        args.max_hops = int(min(max(math.ceil(diam) + 1, max_dist + 1), args.max_hops_cap))
        print(f"auto max_hops = {args.max_hops} (diameter ~{diam}, max s-t distance "
              f"{max_dist}, cap {args.max_hops_cap})", flush=True)
    if max_dist >= args.max_hops:
        raise ValueError(f"max_hops={args.max_hops} must exceed the largest s->t distance "
                         f"{max_dist}: pairs at distance >= max_hops get no hop mask or path.")
    info = register(meta)
    loaders, _ = build_loaders(args.data, args)

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [train_one_seed(args, info, loaders, s) for s in seeds]
    test = np.array([r["test"] for r in runs])
    train = np.array([r["train"] for r in runs])
    print(f"RESULT {os.path.basename(args.data.rstrip('/'))}: "
          f"test acc {test.mean():.4f} ± {test.std():.4f} | "
          f"train acc {train.mean():.4f} ± {train.std():.4f} | seeds {seeds}", flush=True)

    if args.results:
        row = {"dataset": os.path.basename(args.data.rstrip("/")), "meta": meta,
               "args": {k_: v for k_, v in vars(args).items() if k_ != "device"},
               "test_mean": float(test.mean()), "test_std": float(test.std()),
               "train_mean": float(train.mean()), "train_std": float(train.std()),
               "runs": runs}
        with open(args.results, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

