"""
Training script for the Graphormer model on LRGB graph-level tasks.

Uses the HuggingFace-style Graphormer architecture from graphormer_hf/
*without any modifications*. The graph token (position 0 in the sequence)
serves as the graph-level representation for classification / regression.

Datasets: Peptides-func, Peptides-struct.

Examples:
    # Peptides-func (multi-label classification, 10 tasks)
    python train_graphormer.py --dataset Peptides-func

    # Peptides-struct (regression, 11 targets)
    python train_graphormer.py --dataset Peptides-struct

    # Custom hidden dim / heads
    python train_graphormer.py --dataset Peptides-func \\
        --hidden_dim 320 --num_heads 8 --num_layers 6
"""

from __future__ import annotations

from typing import Any

import argparse
import os
import time
import numpy as np
import torch
torch.set_float32_matmul_precision("high")

from torch_geometric.datasets import LRGBDataset
from torch.utils.data import DataLoader as TorchDataLoader

from metrics import build_task
from optim_utils import build_grouped_optimizer_and_scheduler

# Graphormer components
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import preprocess_item


# ------------------------------------------------------------------ #
# Thin wrapper: Graphormer encoder + graph-level head                  #
# ------------------------------------------------------------------ #
import torch.nn as nn
from graphormer_hf.modeling_graphormer import GraphormerGraphEncoder


class GraphormerForGraphClassification(nn.Module):
    """Graphormer encoder with a graph-level classification / regression head.

    Uses the *graph token* (virtual node at position 0) as the graph-level
    representation, following the original Graphormer paper.

    No changes are made to the Graphormer attention or encoder architecture.
    """

    def __init__(self, config: GraphormerConfig, output_dim: int):
        super().__init__()
        self.config = config
        self.graph_encoder = GraphormerGraphEncoder(config)

        # Post-encoder projection (mirrors GraphormerModel)
        from transformers.activations import ACT2FN
        self.lm_head_transform_weight = nn.Linear(
            config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        # Classification / regression head on the graph token
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, output_dim),
        )

    def forward(
        self,
        input_nodes,
        input_edges,
        attn_bias,
        in_degree,
        out_degree,
        spatial_pos,
        attn_edge_type,
    ):
        inner_states, graph_rep, attn_weight = self.graph_encoder(
            input_nodes, input_edges, attn_bias,
            in_degree, out_degree, spatial_pos, attn_edge_type,
        )

        # inner_states[-1]: (seq_len, batch, hidden_dim) — seq includes graph token
        # graph_rep: (batch, hidden_dim) — already the first token embedding
        x = inner_states[-1].transpose(0, 1)  # (B, seq_len, d)

        # Apply LM head transform (matches original GraphormerModel)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # Graph token is at position 0
        graph_token = x[:, 0, :]  # (B, d)

        logits = self.classifier(graph_token)  # (B, output_dim)
        return logits


# ------------------------------------------------------------------ #
# Collator for multi-graph Graphormer batches                          #
# ------------------------------------------------------------------ #

class PeptideGraphormerCollator:
    """Collate pre-processed Graphormer dicts into padded batches.

    Unlike ``GraphormerDataCollator`` this never caches and does not
    assume PyG ``_store`` dicts — it works directly with the output of
    ``preprocess_item`` stored in ``GraphormerPeptideDataset``.
    """

    def __init__(self, spatial_pos_max: int = 20):
        self.spatial_pos_max = spatial_pos_max

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        from collections.abc import Mapping
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        edge_feat_size = len(features[0]["attn_edge_type"][0][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in features)
        edge_input_size = len(features[0]["input_edges"][0][0][0])
        batch_size = len(features)

        batch = {}
        batch["attn_bias"] = torch.zeros(
            batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float)
        batch["attn_edge_type"] = torch.zeros(
            batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch["spatial_pos"] = torch.zeros(
            batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch["in_degree"] = torch.zeros(
            batch_size, max_node_num, dtype=torch.long)
        batch["input_nodes"] = torch.zeros(
            batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch["input_edges"] = torch.zeros(
            batch_size, max_node_num, max_node_num, max_dist,
            edge_input_size, dtype=torch.long)

        for ix, f in enumerate(features):
            # Convert numpy arrays to tensors if needed
            for k in ["attn_bias", "attn_edge_type", "spatial_pos",
                      "in_degree", "input_nodes", "input_edges"]:
                if isinstance(f[k], np.ndarray):
                    f[k] = torch.from_numpy(f[k])
                elif isinstance(f[k], torch.Tensor):
                    f[k] = f[k].detach().clone()

            # Clamp long-range spatial positions to -inf in attn_bias
            sp = f["spatial_pos"]
            ab = f["attn_bias"]
            if (ab[1:, 1:][sp >= self.spatial_pos_max]).numel() > 0:
                ab[1:, 1:][sp >= self.spatial_pos_max] = float("-inf")

            n = f["attn_bias"].shape[0]
            batch["attn_bias"][ix, :n, :n] = f["attn_bias"]
            n2 = f["attn_edge_type"].shape[0]
            batch["attn_edge_type"][ix, :n2, :n2, :] = f["attn_edge_type"]
            batch["spatial_pos"][ix, :n2, :n2] = f["spatial_pos"]
            batch["in_degree"][ix, :f["in_degree"].shape[0]] = f["in_degree"]
            batch["input_nodes"][ix, :f["input_nodes"].shape[0], :] = f["input_nodes"]
            ie = f["input_edges"]
            batch["input_edges"][
                ix, :ie.shape[0], :ie.shape[1], :ie.shape[2], :] = ie

        batch["out_degree"] = batch["in_degree"]

        # Labels
        sample = features[0]["labels"]
        if isinstance(sample, np.ndarray):
            sample_len = len(sample)
        else:
            sample_len = sample.shape[0] if hasattr(sample, 'shape') else len(sample)

        if sample_len == 1:
            batch["labels"] = torch.from_numpy(
                np.concatenate([np.asarray(i["labels"]) for i in features]))
        else:
            batch["labels"] = torch.from_numpy(
                np.stack([np.asarray(i["labels"]) for i in features], axis=0))

        return batch


# ------------------------------------------------------------------ #
# PyG → Graphormer preprocessing dataset wrapper                       #
# ------------------------------------------------------------------ #

class GraphormerPeptideDataset(torch.utils.data.Dataset):
    """Wraps a PyG LRGBDataset and applies Graphormer preprocessing on-the-fly.

    Each item goes through ``preprocess_item`` (Floyd-Warshall shortest
    paths, edge input construction) and is returned as a dict consumable
    by ``GraphormerDataCollator``.
    """

    def __init__(self, pyg_dataset, config, split="train"):
        self.pyg_dataset = pyg_dataset
        self.config = config
        self.split = split
        # Pre-process all items upfront to avoid Cython overhead in
        # DataLoader workers (the preprocessing is deterministic for
        # non-augmented splits).
        # Disable the single-graph CACHED global in collating_graphormer —
        # that cache was designed for node-level tasks with one graph and
        # would make every Peptides graph return the same first result.
        import graphormer_hf.collating_graphormer as _cg
        _cg.CACHED = None

        self.items = []
        for i in range(len(pyg_dataset)):
            _cg.CACHED = None  # reset per-graph
            g = pyg_dataset[i]
            item = self._graph_to_dict(g)
            processed = preprocess_item(item, config=self.config,
                                        keep_features=True, split=self.split)
            self.items.append(processed)

    @staticmethod
    def _graph_to_dict(g):
        """Convert a PyG Data object to the dict format preprocess_item expects."""
        d = {}
        d["x"] = g.x.numpy() if isinstance(g.x, torch.Tensor) else g.x
        d["edge_index"] = (g.edge_index.numpy()
                           if isinstance(g.edge_index, torch.Tensor)
                           else g.edge_index)
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            d["edge_attr"] = (g.edge_attr.numpy()
                              if isinstance(g.edge_attr, torch.Tensor)
                              else g.edge_attr)
        else:
            n_edges = d["edge_index"].shape[1]
            d["edge_attr"] = np.ones((n_edges, 1), dtype=np.int64)
        # Labels
        y = g.y
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        d["labels"] = y
        d["y"] = y
        return d

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ------------------------------------------------------------------ #
# CLI                                                                    #
# ------------------------------------------------------------------ #

def build_parser():
    p = argparse.ArgumentParser(description="Train Graphormer on LRGB")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=["Peptides-func", "Peptides-struct"])

    # Model — Graphormer architecture
    p.add_argument("--hidden_dim", type=int, default=320,
                   help="Embedding / hidden dimension (embedding_dim in config).")
    p.add_argument("--num_heads", type=int, default=8,
                   help="Number of attention heads. head_dim = hidden_dim / num_heads. "
                        "Default 320/8 = 40.")
    p.add_argument("--num_layers", type=int, default=6,
                   help="Number of Graphormer encoder layers.")
    p.add_argument("--ffn_dim", type=int, default=None,
                   help="FFN intermediate dim. Defaults to 4*hidden_dim.")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--attention_dropout", type=float, default=0.2)
    p.add_argument("--activation_dropout", type=float, default=0.1)
    p.add_argument("--multi_hop_max_dist", type=int, default=5,
                   help="Max shortest-path distance for multi-hop edge encoding.")
    p.add_argument("--spatial_pos_max", type=int, default=20,
                   help="Spatial position clamp in collator.")
    p.add_argument("--edge_type", type=str, default="multi_hop",
                   choices=["multi_hop", "single_hop"])

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--reduce_lr_patience", type=int, default=10,
                   help="Patience for ReduceLROnPlateau (epochs).")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_graphormer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a checkpoint .pt file to resume training from.")
    return p


def parse_args():
    args, unknown = build_parser().parse_known_args()
    print(args.__dict__)
    print("Unknown args = ", unknown)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ffn_dim is None:
        args.ffn_dim = 4 * args.hidden_dim
    return args


# ------------------------------------------------------------------ #
# Build model from args + task                                         #
# ------------------------------------------------------------------ #

def build_graphormer_config(args, task):
    """Create a GraphormerConfig for the Peptides datasets."""
    # Peptides node features are 9-dim categorical integers; after
    # preprocess_item (+1 shift) they stay (N, 9).  We pass the raw
    # feature width so GraphormerGraphNodeFeature builds the right Linear.
    from data import get_dataset_info
    ds_info = get_dataset_info(args.dataset)
    node_feat_dim = ds_info["node_feat_dim"]   # 9 for Peptides

    config = GraphormerConfig(
        num_classes=task.output_dim,
        # Node features: peptides have 9 categorical atom features
        num_atoms=512 * 9,
        num_edges=512 * 3,
        num_in_degree=512,
        num_out_degree=512,
        num_spatial=512,
        num_edge_dis=128,
        multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max,
        edge_type=args.edge_type,
        max_nodes=512,
        share_input_output_embed=False,
        num_hidden_layers=args.num_layers,
        embedding_dim=args.hidden_dim,
        ffn_embedding_dim=args.ffn_dim,
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        layerdrop=0.0,
        encoder_normalize_before=True,
        pre_layernorm=True,
        apply_graphormer_init=True,
        activation_fn="gelu",
        embed_scale=None,
        freeze_embeddings=False,
        num_trans_layers_to_freeze=0,
        traceable=False,
        q_noise=0.0,
        qn_block_size=8,
        bias=True,
        # Extra flags consumed by the existing Graphormer code
        dataset_name=args.dataset,
        node_feat_dim=node_feat_dim,
        remove_attn_bias=False,
        enable_spatial_encoder=True,
        enable_diffusion=False,
        enable_layerwise_diffusion=False,
        node_augmentation=False,
        augment_edges=False,
        create_subgraph=False,
    )
    return config


# ------------------------------------------------------------------ #
# Training / evaluation loop                                           #
# ------------------------------------------------------------------ #

def _move_batch_to_device(batch, device):
    """Move the collated Graphormer batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v  # lists (edge_index) stay on CPU
    return out


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None,
              grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds_acc, labels_acc = [], [], []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(
                input_nodes=batch["input_nodes"],
                input_edges=batch["input_edges"],
                attn_bias=batch["attn_bias"],
                in_degree=batch["in_degree"],
                out_degree=batch["out_degree"],
                spatial_pos=batch["spatial_pos"],
                attn_edge_type=batch["attn_edge_type"],
            )
            loss = task.loss(logits, batch["labels"].to(device))

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(batch["labels"]))

    y_pred = np.concatenate(preds_acc, axis=0)
    y_true = np.concatenate(labels_acc, axis=0)
    metric = task.compute_metric(y_pred, y_true)
    return float(np.mean(losses)), float(metric)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    task = build_task(args.dataset)

    print(f"[{args.dataset}] task={task.task_type} level={task.level} "
          f"metric={task.metric_name} (higher_is_better={task.higher_is_better})",
          flush=True)

    # Build Graphormer config
    config = build_graphormer_config(args, task)

    # Load LRGB datasets
    print("Loading datasets...", flush=True)
    train_ds = LRGBDataset(root="./data", name=args.dataset, split="train")
    val_ds = LRGBDataset(root="./data", name=args.dataset, split="val")
    test_ds = LRGBDataset(root="./data", name=args.dataset, split="test")

    # Wrap with Graphormer preprocessing
    print("Preprocessing graphs (Floyd-Warshall + edge encoding)...", flush=True)
    train_processed = GraphormerPeptideDataset(train_ds, config, split="train")
    val_processed = GraphormerPeptideDataset(val_ds, config, split="val")
    test_processed = GraphormerPeptideDataset(test_ds, config, split="test")

    # Collator — custom implementation that works with multi-graph datasets
    collator = PeptideGraphormerCollator(spatial_pos_max=args.spatial_pos_max)

    train_loader = TorchDataLoader(
        train_processed, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator,
    )
    val_loader = TorchDataLoader(
        val_processed, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator,
    )
    test_loader = TorchDataLoader(
        test_processed, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator,
    )

    # Build model
    model = GraphormerForGraphClassification(
        config=config,
        output_dim=task.output_dim,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.3f}M", flush=True)
    print(f"Graphormer config: hidden_dim={args.hidden_dim}, num_heads={args.num_heads}, "
          f"head_dim={args.hidden_dim // args.num_heads}, "
          f"num_layers={args.num_layers}, ffn_dim={args.ffn_dim}", flush=True)

    # Optimizer & scheduler
    total_steps = max(args.max_epochs * len(train_loader), 1)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=list(model.named_parameters()),
        lr_max=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    plateau_mode = "max" if task.higher_is_better else "min"
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=plateau_mode,
        factor=0.5,
        patience=args.reduce_lr_patience,
        min_lr=args.lr_min,
    )
    print(f"ReduceLROnPlateau: mode={plateau_mode}, patience={args.reduce_lr_patience}, "
          f"factor=0.5, min_lr={args.lr_min}", flush=True)

    best_val = -float("inf") if task.higher_is_better else float("inf")
    best_test = None
    best_epoch = -1
    epochs_since_improve = 0
    start_epoch = 0

    # ------------------------------------------------------------------ #
    # Resume from checkpoint                                               #
    # ------------------------------------------------------------------ #
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=args.device)

        model.load_state_dict(ckpt["model"])
        print("  ✓ model weights loaded", flush=True)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  ✓ optimizer state loaded", flush=True)

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("  ✓ step-scheduler state loaded", flush=True)
        elif "epoch" in ckpt:
            completed_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(completed_steps):
                scheduler.step()
            print(f"  ✓ step-scheduler fast-forwarded ({completed_steps} steps)",
                  flush=True)

        if "plateau_scheduler" in ckpt:
            plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
            print("  ✓ plateau-scheduler state loaded", flush=True)

        if "best_val" in ckpt:
            best_val = ckpt["best_val"]
        if "best_test" in ckpt:
            best_test = ckpt["best_test"]
        if "best_epoch" in ckpt:
            best_epoch = ckpt["best_epoch"]
        if "epochs_since_improve" in ckpt:
            epochs_since_improve = ckpt["epochs_since_improve"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

        print(f"  Resuming from epoch {start_epoch} "
              f"(best_val={best_val:.4f} at epoch {best_epoch})", flush=True)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()
        tr_loss, tr_metric = run_epoch(
            model, train_loader, task, args.device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        )
        va_loss, va_metric = run_epoch(model, val_loader, task, args.device)
        plateau_scheduler.step(va_metric)
        te_loss, te_metric = run_epoch(model, test_loader, task, args.device)
        dt = time.time() - t0

        improved = (
            va_metric > best_val if task.higher_is_better else va_metric < best_val
        )
        if improved:
            best_val = va_metric
            best_test = te_metric
            best_epoch = epoch
            epochs_since_improve = 0
            ckpt_path = os.path.join(args.save_dir, f"best_{args.dataset}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "plateau_scheduler": plateau_scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val": best_val,
                    "best_test": best_test,
                    "best_epoch": best_epoch,
                    "epochs_since_improve": epochs_since_improve,
                    "val_metric": va_metric,
                    "test_metric": te_metric,
                },
                ckpt_path,
            )
        else:
            epochs_since_improve += 1

        ml = task.metric_label
        print(
            f"epoch {epoch:03d} | {dt:5.1f}s | "
            f"train loss {tr_loss:.4f} {ml} {tr_metric:.4f} | "
            f"val loss {va_loss:.4f} {ml} {va_metric:.4f} | "
            f"test loss {te_loss:.4f} {ml} {te_metric:.4f} | "
            f"best val {best_val:.4f} (epoch {best_epoch}, test {best_test})",
            flush=True,
        )

        if epochs_since_improve >= args.patience:
            print(f"early stopping at epoch {epoch} "
                  f"(no val improvement in {args.patience} epochs)", flush=True)
            break

    print(f"BEST: val {ml} {best_val:.4f} | test {ml} {best_test:.4f} "
          f"(epoch {best_epoch})", flush=True)


if __name__ == "__main__":
    main()
