"""
Training script for the hop-masked transformer model.

Each attention head is constrained to attend within a chosen hop range,
preserving per-hop identity in a transformer-native way. This addresses
the empirical observation that approaches collapsing the K hop axis
(LRU, proxy-bottleneck routing, GNN-pooling) underperform the simple
concat baseline.

Datasets: Peptides-func, Peptides-struct, PascalVOC-SP, MNIST, CIFAR10,
PATTERN, CLUSTER, ZINC12k.

Examples:
    # Default: contiguous partition of hops [1..K-1] across heads.
    python train_hop_masked_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --num_layers 4 --hop_mode contiguous

    # Window mode: each head sees ~3 adjacent hops, centres spread evenly.
    python train_hop_masked_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --hop_mode window --hop_window 1

    # Mix structured heads with unrestricted "global" heads.
    python train_hop_masked_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --num_global_heads 2

    # PascalVOC-SP — smaller batch and shorter hop horizon.
    python train_hop_masked_transformer.py --dataset PascalVOC-SP \\
        --max_hops 12 --num_heads 4 --num_layers 4 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

from data import DATASET_CHOICES, get_loaders
from metrics import build_task, compute_pos_weight, compute_class_weights
from model_hop_masked_transformer_final_4 import (HopMaskedTransformerModel,
                                                set_attn_diagnostics)
from optim_utils import build_grouped_optimizer_and_scheduler

os.environ["PYTHON_HASH_SEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _gather_train_labels(loader, num_classes):
    """Collect the (N, C) multi-label target matrix from a training loader."""
    ds = loader.dataset
    base = getattr(ds, "pyg_dataset", ds)   # unwrap DistMaskDataset
    rows = []
    for i in range(len(base)):
        g = base[i]
        y = g.y
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        rows.append(y.reshape(-1)[:num_classes])
    return np.stack(rows, axis=0)


def _gather_node_labels(loader, num_classes):
    """Collect all per-node class indices from a training loader (multiclass).

    Returns a 1-D int array of every node's label across all training graphs,
    with padding (-1) removed.
    """
    ds = loader.dataset
    base = getattr(ds, "pyg_dataset", ds)   # unwrap DistMaskDataset
    rows = []
    for i in range(len(base)):
        g = base[i]
        y = g.y
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        rows.append(y.reshape(-1))
    labels = np.concatenate(rows, axis=0)
    return labels[labels >= 0]


def build_pos_or_class_weight(args, dataset_info, train_loader, device):
    """Compute the loss weight vector when --use_pos_weight is set.

    - multi_label -> per-class BCE pos_weight  = sqrt(N/(C*n_k))
    - multiclass  -> per-class CE class weights = N/(C*n_c), mean-normalised
                     (routes class weights to node/graph multiclass tasks)
    Returns a (C,) float tensor on ``device`` or None.
    """
    if not args.use_pos_weight:
        return None
    ttype = dataset_info["task_type"]
    C = dataset_info["output_dim"]
    if ttype == "multi_label":
        w = compute_pos_weight(_gather_train_labels(train_loader, C), C)
        print(f"pos_weight (sqrt(N/(C*n_k))): {np.round(w, 3)}", flush=True)
    elif ttype == "multiclass":
        w = compute_class_weights(_gather_node_labels(train_loader, C), C)
        print(f"class_weight N/(C*n_c) mean-norm: {np.round(w, 3)}", flush=True)
    else:
        print(f"--use_pos_weight ignored: task_type={ttype} unsupported.", flush=True)
        return None
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def build_parser():
    p = argparse.ArgumentParser(description="Train hop-masked transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=DATASET_CHOICES)
    p.add_argument("--max_hops", type=int, default=40,
                   help="K — total hop levels in the precomputed dist_masks. "
                        "Use ~12 for PascalVOC-SP/COCO-SP and ~8 for the "
                        "transductive datasets (the collate buffer is "
                        "B*max_hops*N^2*4 bytes).")
    # ---- Transductive subgraph sampling (ogbn-arxiv, ogbn-products,
    # ---- arxiv-year only; ignored for every other dataset) ----------------
    p.add_argument("--subgraph_mode", type=str, default="partition",
                   choices=["partition", "egonet"],
                   help="How to cut a single huge graph into trainable pieces. "
                        "'partition' mirrors S2GNN (random node partition, "
                        "induced subgraphs). 'egonet' takes a k-hop "
                        "neighbourhood per labelled node.")
    p.add_argument("--num_parts", type=int, default=128,
                   help="subgraph_mode=partition: number of random parts. "
                        "Target 1000-2000 nodes/part (arxiv ~128, "
                        "products ~2048).")
    p.add_argument("--egonet_hops", type=int, default=2,
                   help="subgraph_mode=egonet: neighbourhood radius.")
    p.add_argument("--egonet_max_nodes", type=int, default=1024,
                   help="subgraph_mode=egonet: cap on nodes per ego net.")
    p.add_argument("--max_egonet_samples", type=int, default=None,
                   help="subgraph_mode=egonet: subsample this many seed nodes "
                        "per split (None = every labelled node).")
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)
    p.add_argument("--mask_type", type=str, default="shortest_path",
                   choices=["shortest_path", "adj_power"],
                   help="How hop masks are built. 'shortest_path' (default) "
                        "uses the precomputed distance shells. 'adj_power' "
                        "rebuilds masks from powers of the adjacency: slot k = "
                        "(A^k > 0), reconstructing A from the distance-1 shell. "
                        "Pair with --adj_self_loops to use (A+I)^k.")
    p.add_argument("--adj_self_loops", action="store_true", default=False,
                   help="Only with --mask_type adj_power: use (A+I)^k instead "
                        "of A^k. (A+I)^k means 'reachable in <= k steps' "
                        "(monotone, no parity striping on near-bipartite "
                        "graphs); A^k is exact-length-k walks.")

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8,
                   help="Total heads. Must divide hidden_dim.")
    p.add_argument("--ffn_ratio", type=float, default=1)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--graph_pool", type=str, default="sum",
                   choices=["sum", "mean", "attention"],
                   help="Graph-level readout. 'attention' uses a learnable "
                        "query attending over nodes (Set-Transformer PMA).")
    p.add_argument("--norm_type", type=str, default="layer",
                   choices=["layer", "rms", "graph"],
                   help="Normalization in transformer layers. 'layer' "
                        "(default) = LayerNorm; 'rms' = RMSNorm; 'graph' = "
                        "masked GraphNorm (per-graph statistics).")
    p.add_argument("--v_head_dim", type=int, default=None,
                   help="Value head dim. Default None = equals QK head dim "
                        "(hidden_dim // num_heads, original behaviour). Set "
                        "larger to decouple value capacity from QK "
                        "(asymmetric attention).")
    p.add_argument("--block_diag_out", action="store_true", default=False,
                   help="Use block-diagonal out_proj in MHA (no cross-head "
                        "mixing inside attention).")
    p.add_argument("--dynamic_cross_hop", action="store_true", default=False,
                   help="Insert a dynamic cross-hop attention sublayer "
                        "between MHA and FFN. Best paired with "
                        "--block_diag_out.")
    p.add_argument("--cross_hop_hop_embedding", action="store_true", default=False,
                   help="Add a learnable per-head (per-hop-band) embedding "
                        "inside the DynamicCrossHopMixer so it is no longer "
                        "permutation-invariant over hop slabs. Only has an "
                        "effect when --dynamic_cross_hop is set.")
    p.add_argument("--multihop_attn", action="store_true", default=False,
                   help="Use multi-hop masked attention: every head is masked "
                        "by every hop (H × K views) plus an optional global "
                        "view, sharing one q·kᵀ per head.  The K per-hop "
                        "views are collapsed back to hidden_dim by a sum/mean "
                        "readout (--multihop_readout).  Decouples heads from a "
                        "fixed hop assignment; mutually exclusive with "
                        "--use_moe_gating.  When active, --blend_adj_power, "
                        "--use_edge_bias, and --use_rrwp have no effect.")
    p.add_argument("--multihop_readout", type=str, default="sum",
                   choices=["sum", "mean"],
                   help="How to collapse the per-hop attention views back to "
                        "hidden_dim in --multihop_attn.  'mean' divides per "
                        "query node by the number of non-empty hop views, "
                        "preventing diameter-dependent magnitude scaling.")
    p.add_argument("--multihop_no_global", action="store_true", default=False,
                   help="Drop the unmasked global view in --multihop_attn. "
                        "By default each head also computes one unmasked "
                        "(padding-only) attention output on top of the K "
                        "hop-masked outputs.")
    p.add_argument("--use_edge_features", action="store_true", default=False,
                   help="Incorporate edge/bond features: a BondEncoder embeds "
                        "edge_attr, aggregated into node features in the node "
                        "encoder, and (if --num_post_gat_layers>0) fed to the "
                        "GATv2 layers via edge_dim.")
    p.add_argument("--use_pos_weight", action="store_true", default=False,
                   help="Use per-class pos_weight = sqrt(N/(C*n_k)) in the BCE "
                        "loss for multi-label tasks (computed from the training "
                        "set). No effect on non-multi-label tasks.")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal loss focusing parameter γ ≥ 0.  0 (default) = "
                        "standard BCE / CE loss.  Positive values apply focal "
                        "down-weighting (1-p_t)^γ to easy examples.  Typical: "
                        "0.5, 1, 2.  Compatible with --use_pos_weight (the "
                        "pos_weight acts as per-class alpha on top of focal "
                        "weighting).  No effect on regression tasks.")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing epsilon ε ≥ 0.  0 (default) = hard "
                        "targets.  For multi_label tasks replaces {0,1} targets "
                        "with {ε, 1−ε} before BCE, preventing the model from "
                        "driving training loss to zero on confident positives and "
                        "compressing the train-val confidence gap.  For multiclass "
                        "tasks passes ε to CrossEntropyLoss natively (not "
                        "supported alongside --focal_gamma for multiclass).  "
                        "No effect on regression.  Typical values: 0.05–0.1.")
    p.add_argument("--use_virtual_node", action="store_true", default=False,
                   help="Prepend a learnable virtual-node embedding that "
                        "participates in every attention head.  For graph-"
                        "level tasks its final embedding is used as the "
                        "graph representation (replaces pooling).")
    p.add_argument("--num_post_gat_layers", type=int, default=0,
                   help="Number of GATv2Conv layers applied after the "
                        "transformer stack, before the task head.  "
                        "0 = disabled (default).")
    p.add_argument("--num_gat_heads", type=int, default=4,
                   help="Number of attention heads in each post-transformer "
                        "GATv2 layer.  Must divide hidden_dim.")

    # New attention bias features
    p.add_argument("--cross_hop_no_ffn", action="store_true", default=False,
                   help="When dynamic_cross_hop is enabled, skip the FFN after "
                        "the cross-hop mixer (FFN becomes redundant).")
    p.add_argument("--blend_adj_power", action="store_true", default=False,
                   help="Enable learnable blending of adjacency-power walk-count "
                        "scores as attention bias within each head's hop mask. "
                        "Adds per-head learnable scalar (adj_blend_gamma).")
    p.add_argument("--use_edge_bias", action="store_true", default=False,
                   help="Enable edge feature attention bias for hop-1 heads. "
                        "Uses BondEncoder to embed bond features, then projects "
                        "to per-head scalar biases in attention scores.")
    p.add_argument("--use_rrwp", action="store_true", default=False,
                   help="Enable Relative Random Walk Probability (RRWP) as "
                        "attention bias. Computed on-the-fly from adjacency "
                        "matrix. Applied to all heads.")
    p.add_argument("--rrwp_dim", type=int, default=8,
                   help="Number of random-walk steps (powers of P = D^{-1}A) "
                        "to compute for RRWP bias. Only used when --use_rrwp "
                        "is set.")

    # Hop-to-head assignment
    p.add_argument("--hop_mode", type=str, default="contiguous",
                   choices=["contiguous", "window", "single", "interleaved",
                            "alternating", "file"],
                   help="How to assign hops to heads. contiguous = partition "
                        "[1..K-1] into num_heads chunks; window = evenly-spaced "
                        "centres with hop_window half-width; single = exactly "
                        "one hop per head (requires num_heads == K-1); "
                        "interleaved = each head covers every hop_window-th hop "
                        "starting from an evenly-spaced centre k; "
                        "alternating = even-indexed layers get even hops, "
                        "odd-indexed layers get odd hops.")
    p.add_argument("--hop_window", type=int, default=1,
                   help="Half-window for 'window' mode (head covers "
                        "[c-w, c+w]).")
    p.add_argument("--num_global_heads", type=int, default=0,
                   help="Last N heads are unrestricted (free global "
                        "attention) rather than hop-masked.")
    p.add_argument("--hop_file", type=str, default=None,
                   help="Path to a JSON file for 'file' hop_mode. Keys are "
                        "1-indexed head numbers, values are lists of hop "
                        "indices. E.g. {\"1\": [0, 1], \"2\": [0, 2, 4]}. "
                        "Heads not listed get global attention.")

    # MoE gating (optional, replaces deterministic hop assignment)
    p.add_argument("--use_moe_gating", action="store_true", default=False,
                   help="Replace deterministic hop-to-head assignment with a "
                        "learned MoE gating network. Each head dynamically "
                        "selects which hop masks to attend through.")
    p.add_argument("--top_k", type=int, default=0,
                   help="Sparse gating: each head keeps only top-k hops. "
                        "0 = dense (full softmax over all K hops). "
                        "Only used when --use_moe_gating is set.")
    p.add_argument("--gate_noise", type=float, default=0.1,
                   help="Gaussian noise std added to gate logits during "
                        "training to encourage exploration. "
                        "Only used when --use_moe_gating is set.")
    p.add_argument("--balance_coeff", type=float, default=0.01,
                   help="Weight for Switch-style load-balancing aux loss. "
                        "Only used when --use_moe_gating is set.")
    p.add_argument("--entropy_coeff", type=float, default=0.01,
                   help="Weight for entropy regularisation aux loss "
                        "(encourages diffuse gate distributions). "
                        "Only used when --use_moe_gating is set.")

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--reduce_lr_patience", type=int, default=10,
                   help="Patience for ReduceLROnPlateau (epochs).")
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Diagnostics
    p.add_argument("--log_head_stats_interval", type=int, default=10,
                   help="Print per-head diagnostic stats every N validation "
                        "epochs (0 = disabled).  For MoE: logs gate-weight "
                        "distribution (which hops each head specialises on). "
                        "For blend_adj_power: logs adj_blend_gamma per head.")

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_hop_masked")
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
    return args


def _move_batch_to_device(batch, device):
    pyg_batch, dist_masks, node_masks = batch
    return pyg_batch.to(device), dist_masks.to(device), node_masks.to(device)


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None,
              grad_clip=1.0, collect_gate_weights=False):
    """Run one epoch.

    Returns:
        (loss, metric, gate_weights_mean_or_None)

    gate_weights_mean_or_None — non-None only when ``collect_gate_weights=True``
        and the model uses MoE gating.  Shape: (H, K), CPU tensor, the per-head
        gate-probability distribution averaged over all batches in this epoch
        (and over layers).  Each row sums to ~1 (softmax).
    """
    is_train = optimizer is not None
    model.train(is_train)

    # Gate-weight accumulator: running sum of per-batch (H, K) means.
    _gw_sum = None    # (H, K) CPU tensor
    _gw_count = 0
    # Only collect on eval passes to avoid any gradient/memory overhead on train.
    _do_collect = collect_gate_weights and (not is_train)

    losses, preds_acc, labels_acc = [], [], []
    for batch in loader:
        pyg_batch, dist_masks, node_masks = _move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, _, aux_loss, gw = model(
                pyg_batch, dist_masks, node_masks,
                return_gate_weights=_do_collect,
            )
            task_loss = task.loss(logits, pyg_batch.y)
            loss = task_loss + aux_loss

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(task_loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(pyg_batch.y))

        # Accumulate gate weights (running mean — keeps only (H,K) in memory).
        if _do_collect and gw is not None:
            # gw: (H, K) CPU, already batch-averaged inside model.forward
            _gw_sum  = gw if _gw_sum is None else _gw_sum + gw
            _gw_count += 1

    y_pred = np.concatenate(preds_acc, axis=0)
    y_true = np.concatenate(labels_acc, axis=0)
    metric = task.compute_metric(y_pred, y_true)
    gate_weights_mean = (_gw_sum / _gw_count) if _gw_count > 0 else None
    return float(np.mean(losses)), float(metric), gate_weights_mean


def _log_head_stats(epoch, model, gate_weights_mean, args):
    """Print per-head diagnostic statistics.

    Two sections (each only printed when the relevant feature is active):

    MoE gate distribution
        For each head: entropy + top-5 hops by probability, averaged over
        the validation set and all layers.  Lets you see which heads
        specialise on which hop bands and how diffuse they are.

    Adj-blend gammas
        For each head: the raw learnable gamma_h scalar.  Positive values
        mean the model is boosting walk-count topology within the hop mask;
        negative values suppress it.  Displayed alongside the hop assignment
        so the chemical/structural interpretation is immediate.
        With multiple layers each layer's gammas are shown separately.
    """
    prefix = f"[epoch {epoch:03d}]"

    # ── Attention diagnostics ─────────────────────────────────────────────────
    # Populated only when set_attn_diagnostics(True) was active during the
    # preceding val pass. Reads whatever the last val batch left in _diag.
    #
    #   ent_n   normalised entropy, 1.0 = uniform over allowed keys, 0.0 = one key
    #   part    participation ratio = effective number of keys attended
    #   keys    mean allowed keys per query row (structural capacity of the hop)
    #   self    attention mass staying on the query node
    #   |out|   norm of what the head transports (near 0 = head moves nothing
    #           even if its attention looks healthy)
    #   live    sanity check: rows with any allowed key. 1.0 whenever every
    #           head's hop set contains hop 0; below 1.0 flags a masking bug
    #   live_ns rows with a NON-SELF allowed key, i.e. rows where this hop
    #           actually exists. Every column above is averaged over ONLY those
    #           rows, so a low live_ns means a small effective sample
    for layer_idx, layer in enumerate(getattr(model, "layers", [])):
        d = getattr(getattr(layer, "attn", None), "_diag", None)
        if d is None:
            continue
        hop_sets = getattr(model, "head_hop_sets", None)
        print(f"{prefix} [layer {layer_idx}] hop-attention per head:", flush=True)
        for h in range(d["entropy"].numel()):
            hop_lbl = "global"
            if hop_sets is not None and h < len(hop_sets):
                hs = hop_sets[h]
                hop_lbl = "global" if hs is None else ",".join(map(str, hs))
            print(f"  head {h:02d} | hops={hop_lbl:<12s} "
                  f"ent_n={float(d['entropy_norm'][h]):.3f} "
                  f"part={float(d['participation'][h]):6.2f} "
                  f"keys={float(d['allowed_keys'][h]):6.2f} "
                  f"self={float(d['self_mass'][h]):.3f} "
                  f"|out|={float(d['out_norm'][h]):.3f} "
                  f"live={float(d['live_frac'][h]):.3f} "
                  f"live_ns={float(d['live_nonself'][h]):.3f}", flush=True)

        cd = getattr(getattr(layer, "cross_hop", None), "_diag", None)
        if cd is not None:
            mix = cd["mix_matrix"]
            print(f"{prefix} [layer {layer_idx}] cross-hop mixing "
                  f"(row=query head, col=key head):", flush=True)
            for h in range(mix.shape[0]):
                row = " ".join(f"{float(x):.2f}" for x in mix[h])
                print(f"  head {h:02d} | ent={float(cd['entropy'][h]):.3f} | {row}",
                      flush=True)

    # ── MoE gate distribution ─────────────────────────────────────────────────
    if args.use_moe_gating and gate_weights_mean is not None:
        H, K = gate_weights_mean.shape
        print(f"{prefix} MoE gate distributions (val mean, top-5 hops per head):",
              flush=True)
        for h in range(H):
            probs = gate_weights_mean[h]                           # (K,)
            top_n = min(5, K)
            topk_vals, topk_idx = probs.topk(top_n)
            hop_str = "  ".join(
                f"k={int(topk_idx[i]):02d}({float(topk_vals[i]):.3f})"
                for i in range(top_n)
            )
            ent = float(-(probs * (probs + 1e-9).log()).sum())
            # Argmax hop
            argmax_hop = int(probs.argmax())
            print(f"  head {h:02d} | argmax k={argmax_hop:02d} | ent={ent:.3f} | {hop_str}",
                  flush=True)

    # ── Adj-blend gammas ──────────────────────────────────────────────────────
    if args.blend_adj_power and not args.use_moe_gating:
        hop_sets   = model.head_hop_sets
        num_layers = len(model.layers)
        print(f"{prefix} Adj-blend gammas (gamma>0 = walk-count boosts attn; "
              f"gamma<0 = suppresses):", flush=True)

        for layer_idx, layer in enumerate(model.layers):
            if not hasattr(layer.attn, 'adj_blend_gamma'):
                continue
            gammas = layer.attn.adj_blend_gamma.detach().cpu()   # (H,)
            layer_label = f"  [layer {layer_idx}]" if num_layers > 1 else " "

            # Build compact per-head strings, printed 8 per line
            parts = []
            for h, hop_set in enumerate(hop_sets):
                if hop_set is None:
                    tag = "GLB"
                else:
                    non_self = [k for k in hop_set if k != 0]
                    tag = f"k{non_self[0]:02d}" if non_self else "k00"
                g = float(gammas[h])
                parts.append(f"h{h:02d}[{tag}]:{g:+.3f}")

            if layer_label.strip():
                print(f"{layer_label}", flush=True)
            # Print 8 entries per line for readability
            for i in range(0, len(parts), 8):
                print("    " + "  ".join(parts[i:i + 8]), flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, test_loader, _, _, _, dataset_info = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=True,
        max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe,
        lap_pe_dim=args.lap_pe_dim,
        dataset_name=args.dataset,
        return_info=True,
        subgraph_mode=args.subgraph_mode,
        num_parts=args.num_parts,
        egonet_hops=args.egonet_hops,
        egonet_max_nodes=args.egonet_max_nodes,
        max_egonet_samples=args.max_egonet_samples,
        seed=args.seed,
    )
    dataset_name = dataset_info["name"]

    # Optional loss weighting: multi_label -> BCE pos_weight; multiclass ->
    # inverse-frequency CE class weights (routed to node/graph multiclass).
    pos_weight = build_pos_or_class_weight(args, dataset_info, train_loader, args.device)

    task = build_task(dataset_name, dataset_info=dataset_info, pos_weight=pos_weight,
                      focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    # Move the loss module (and its pos_weight buffer) onto the device.
    task.loss_fn = task.loss_fn.to(args.device)

    _ls_tag = f"+LS(ε={args.label_smoothing})" if args.label_smoothing > 0 else ""
    _loss_name = (
        f"FocalBCE(γ={args.focal_gamma}){_ls_tag}" if args.focal_gamma > 0 and task.task_type == "multi_label"
        else f"FocalCE(γ={args.focal_gamma})" if args.focal_gamma > 0 and task.task_type == "multiclass"
        else f"BCE{_ls_tag}" if task.task_type == "multi_label"
        else f"CE{_ls_tag}" if task.task_type == "multiclass"
        else "L1"
    )
    print(f"[{dataset_name}] task={task.task_type} level={task.level} "
          f"metric={task.metric_name} (higher_is_better={task.higher_is_better}) "
          f"loss={_loss_name}",
          flush=True)

    model = HopMaskedTransformerModel(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_hops=args.max_hops,
        hop_mode=args.hop_mode,
        hop_window=args.hop_window,
        hop_file=args.hop_file,
        num_global_heads=args.num_global_heads,
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=dataset_name,
        node_feat_dim=dataset_info.get("node_feat_dim"),
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop,
        norm_type=args.norm_type,
        v_head_dim=args.v_head_dim,
        mask_type=args.mask_type,
        adj_self_loops=args.adj_self_loops,
        use_moe_gating=args.use_moe_gating,
        top_k=args.top_k,
        gate_noise=args.gate_noise,
        balance_coeff=args.balance_coeff,
        entropy_coeff=args.entropy_coeff,
        use_virtual_node=args.use_virtual_node,
        num_post_gat_layers=args.num_post_gat_layers,
        num_gat_heads=args.num_gat_heads,
        cross_hop_hop_embedding=args.cross_hop_hop_embedding,
        use_edge_features=args.use_edge_features,
        edge_feat_dim=dataset_info.get("edge_feat_dim"),
        cross_hop_no_ffn=args.cross_hop_no_ffn,
        blend_adj_power=args.blend_adj_power,
        use_edge_bias=args.use_edge_bias,
        use_rrwp=args.use_rrwp,
        rrwp_dim=args.rrwp_dim,
        multihop_attn=args.multihop_attn,
        multihop_readout=args.multihop_readout,
        multihop_include_global=not args.multihop_no_global,
        embed_dropout=args.dropout
    ).to(args.device)

    # Print the head -> hop-set assignment so it's logged for reproducibility.
    if args.use_moe_gating:
        print(f"MoE gating config: top_k={args.top_k}, gate_noise={args.gate_noise}, "
              f"balance_coeff={args.balance_coeff}, entropy_coeff={args.entropy_coeff}",
              flush=True)
    elif args.multihop_attn:
        print(f"Multi-hop attention: every head masked by every hop "
              f"(H x K views), readout={args.multihop_readout}, "
              f"global_view={'OFF' if args.multihop_no_global else 'ON'}",
              flush=True)
    else:
        print("Head -> hop set assignment:", flush=True)
        for h, s in enumerate(model.head_hop_sets):
            print(f"  head {h}: {'GLOBAL (no hop mask)' if s is None else s}", flush=True)
    if args.hop_mode == "file":
        print(f"Loaded head->hop assignment from file: {args.hop_file}", flush=True)
    if args.use_virtual_node:
        print("Virtual node: ENABLED (learnable embedding, visible to all heads)",
              flush=True)
    if args.hop_mode == "alternating":
        print("Alternating hop mode:", flush=True)
        for i, hop_sets in enumerate(model.per_layer_hop_sets):
            label = "even" if i % 2 == 0 else "odd"
            print(f"  layer {i} ({label}):", flush=True)
            for h, s in enumerate(hop_sets):
                print(f"    head {h}: {'GLOBAL' if s is None else s}", flush=True)
    if args.num_post_gat_layers > 0:
        print(f"Post-transformer GATv2: {args.num_post_gat_layers} layer(s), "
              f"{args.num_gat_heads} heads", flush=True)
    if args.dynamic_cross_hop and args.cross_hop_hop_embedding:
        print("Cross-hop mixer: per-head hop embedding ENABLED", flush=True)
    if args.use_edge_features:
        print("Edge features: ENABLED (BondEncoder in node encoder"
              + (" + GATv2" if args.num_post_gat_layers > 0 else "") + ")", flush=True)

    # New attention bias features logging
    if args.cross_hop_no_ffn:
        print("Cross-hop no-FFN mode: ENABLED (skip FFN after cross-hop mixer)",
              flush=True)
    if args.blend_adj_power:
        print("Adj-power blend: ENABLED (learnable per-head gamma)", flush=True)
    if args.use_edge_bias:
        print("Edge feature attention bias: ENABLED (hop-1 heads)", flush=True)
    if args.use_rrwp:
        print(f"RRWP bias: ENABLED (dim={args.rrwp_dim})", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.3f}M", flush=True)

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
        # verbose=True,
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

        # Model weights (required)
        model.load_state_dict(ckpt["model"])
        print("  ✓ model weights loaded", flush=True)

        # Optimizer state
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  ✓ optimizer state loaded", flush=True)

        # Per-step cosine scheduler — fast-forward to the saved epoch
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("  ✓ step-scheduler state loaded", flush=True)
        elif "epoch" in ckpt:
            # Fall back: replay the correct number of steps
            completed_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(completed_steps):
                scheduler.step()
            print(f"  ✓ step-scheduler fast-forwarded ({completed_steps} steps)",
                  flush=True)

        # ReduceLROnPlateau scheduler
        if "plateau_scheduler" in ckpt:
            plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
            print("  ✓ plateau-scheduler state loaded", flush=True)

        # Training bookkeeping
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

    # Pre-compute whether head stats should be logged this epoch.
    _log_interval = args.log_head_stats_interval
    # Attention diagnostics need no extra flags, so any positive interval now
    # produces per-head output. The gate/gamma sections still self-gate below.
    _want_head_stats = _log_interval > 0

    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()
        tr_loss, tr_metric, _ = run_epoch(
            model, train_loader, task, args.device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        )
        # Collect gate weights on val pass when it's a logging epoch.
        _collect_gw = _want_head_stats and (epoch % _log_interval == 0)
        if _collect_gw:
            set_attn_diagnostics(True)
        va_loss, va_metric, _gw_mean = run_epoch(
            model, val_loader, task, args.device,
            collect_gate_weights=_collect_gw,
        )
        set_attn_diagnostics(False)
        plateau_scheduler.step(va_metric)
        te_loss, te_metric, _ = run_epoch(model, test_loader, task, args.device)
        dt = time.time() - t0

        improved = (
            va_metric > best_val if task.higher_is_better else va_metric < best_val
        )
        if improved:
            best_val = va_metric
            best_test = te_metric
            best_epoch = epoch
            epochs_since_improve = 0
            ckpt_path = os.path.join(args.save_dir, f"best_{dataset_name}.pt")
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

        # ── Per-head diagnostic logging ───────────────────────────────
        if _collect_gw:
            _log_head_stats(epoch, model, _gw_mean, args)

        if epochs_since_improve >= args.patience:
            print(f"early stopping at epoch {epoch} "
                  f"(no val improvement in {args.patience} epochs)", flush=True)
            break

    print(f"BEST: val {ml} {best_val:.4f} | test {ml} {best_test:.4f} "
          f"(epoch {best_epoch})", flush=True)

    # ------------------------------------------------------------------ #
    # Post-training: error-by-diameter analysis                            #
    # ------------------------------------------------------------------ #
    print("\n=== Error-by-diameter analysis ===", flush=True)
    ckpt_path = os.path.join(args.save_dir, f"best_{dataset_name}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?')})", flush=True)
    else:
        print("No best checkpoint found, using final model weights.", flush=True)

    _plot_error_by_diameter(
        model, train_loader, val_loader, test_loader, task, args.device,
        dataset_name, args.save_dir,
    )


def _get_diameters_from_loader(loader):
    """Extract per-graph **true** diameter from the underlying PyG graphs.

    Computes the all-pairs shortest path via Floyd-Warshall on each graph's
    adjacency matrix and returns the maximum finite distance (the diameter).
    This is independent of the ``max_hops`` truncation applied to the
    distance masks, so diameters beyond ``max_hops`` are reported correctly.
    """
    from scipy.sparse.csgraph import floyd_warshall as fw

    ds = loader.dataset  # DistMaskDataset
    base = getattr(ds, "pyg_dataset", ds)  # unwrap to raw PyG dataset
    diameters = []
    for i in range(len(base)):
        g = base[i]
        n = g.x.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)
        ei = g.edge_index.numpy()
        adj[ei[0], ei[1]] = 1.0
        dist = fw(adj, directed=False, unweighted=True)
        finite = dist[np.isfinite(dist)]
        diam = int(finite.max()) if finite.size > 0 else 0
        diameters.append(diam)
    return np.array(diameters, dtype=np.int32)


@torch.no_grad()
def _collect_per_sample_data(model, loader, task, device):
    """Run model over *loader* and return a 1-D array of per-sample losses,
    along with true labels and predictions (for AP calculation if applicable).

    Works for graph-level tasks (multi_label, regression, multiclass).
    For node-level tasks, the loss is averaged over valid nodes within each
    graph — the result is still one scalar per graph.
    """
    import torch.nn.functional as F

    model.eval()
    per_sample_losses = []
    y_trues = []
    y_preds = []

    for batch in loader:
        pyg_batch, dist_masks, node_masks = _move_batch_to_device(batch, device)
        logits, _, aux_loss, _gw = model(pyg_batch, dist_masks, node_masks)

        # --- Per-sample loss (unreduced) ---
        if task.task_type == "multi_label":
            y = pyg_batch.y.float()
            # (B, C) -> mean over C -> (B,)
            loss_per = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            ).mean(dim=-1)

            y_trues.append(y.detach().cpu().numpy())
            y_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        elif task.task_type == "regression":
            y = pyg_batch.y.float()
            # (B, K) -> mean over K -> (B,)
            loss_per = F.l1_loss(logits, y, reduction="none")
            if loss_per.dim() > 1:
                loss_per = loss_per.mean(dim=-1)
        elif task.task_type == "multiclass":
            y = pyg_batch.y.long()
            if y.dim() > 1:
                y = y.view(-1)
            if task.level == "node":
                # Node-level: aggregate per-node losses back to per-graph
                loss_all = F.cross_entropy(logits, y, ignore_index=-1,
                                           reduction="none")  # (total_nodes,)
                # Use pyg_batch.batch to group nodes -> graphs
                batch_ids = pyg_batch.batch  # (total_nodes,)
                B = int(batch_ids.max().item()) + 1
                graph_losses = []
                for g in range(B):
                    mask = (batch_ids == g) & (y != -1)
                    if mask.any():
                        graph_losses.append(loss_all[mask].mean().item())
                    else:
                        graph_losses.append(0.0)
                loss_per = torch.tensor(graph_losses)
            else:
                loss_per = F.cross_entropy(logits, y, reduction="none")  # (B,)
        else:
            raise ValueError(f"Unknown task_type: {task.task_type}")

        per_sample_losses.append(loss_per.detach().cpu().numpy())

    losses_out = np.concatenate(per_sample_losses, axis=0)
    if task.task_type == "multi_label" and len(y_trues) > 0:
        return losses_out, np.concatenate(y_trues, axis=0), np.concatenate(y_preds, axis=0)
    return losses_out, None, None


def _plot_error_by_diameter(model, train_loader, val_loader, test_loader,
                            task, device, dataset_name, save_dir):
    """Compute per-sample loss, group by graph diameter, and generate box plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Collect diameters ---
    print("  Extracting graph diameters...", flush=True)
    train_diameters = _get_diameters_from_loader(train_loader)
    val_diameters = _get_diameters_from_loader(val_loader)
    test_diameters = _get_diameters_from_loader(test_loader)

    # --- Collect per-sample losses and data ---
    print("  Computing per-sample losses (train)...", flush=True)
    train_losses, train_y, train_p = _collect_per_sample_data(model, train_loader, task, device)
    print("  Computing per-sample losses (val)...", flush=True)
    val_losses, val_y, val_p = _collect_per_sample_data(model, val_loader, task, device)
    print("  Computing per-sample losses (test)...", flush=True)
    test_losses, test_y, test_p = _collect_per_sample_data(model, test_loader, task, device)

    # Verify alignment
    assert len(train_diameters) == len(train_losses), \
        f"Train mismatch: {len(train_diameters)} diameters vs {len(train_losses)} losses"
    assert len(val_diameters) == len(val_losses), \
        f"Val mismatch: {len(val_diameters)} diameters vs {len(val_losses)} losses"
    assert len(test_diameters) == len(test_losses), \
        f"Test mismatch: {len(test_diameters)} diameters vs {len(test_losses)} losses"

    def _safe_ap(y_t, y_p):
        from sklearn.metrics import average_precision_score
        if len(y_t) == 0: return float("nan")
        aps = []
        for c in range(y_t.shape[1]):
            if y_t[:, c].sum() > 0:
                try:
                    aps.append(average_precision_score(y_t[:, c], y_p[:, c]))
                except:
                    pass
        if not aps: return float("nan")
        return np.mean(aps)

    has_ap = task.task_type == "multi_label"

    # --- Print summary statistics per diameter ---
    all_diameters = np.unique(np.concatenate(
        [train_diameters, val_diameters, test_diameters]))

    if has_ap:
        print(f"\n  {'Diam':>5s} | {'Train N':>8s} {'Tr Mean':>9s} {'Tr Med':>8s} {'Tr AP':>8s} "
              f"| {'Val N':>6s} {'Val Mean':>9s} {'Val Med':>8s} {'Val AP':>8s} "
              f"| {'Test N':>7s} {'Te Mean':>9s} {'Te Med':>8s} {'Te AP':>8s}", flush=True)
        print("  " + "-" * 135, flush=True)
    else:
        print(f"\n  {'Diam':>5s} | {'Train N':>8s} {'Train Mean':>11s} {'Train Med':>10s} "
              f"| {'Val N':>6s} {'Val Mean':>9s} {'Val Med':>8s} "
              f"| {'Test N':>7s} {'Test Mean':>10s} {'Test Med':>9s}", flush=True)
        print("  " + "-" * 105, flush=True)

    for d in sorted(all_diameters):
        tr_mask = train_diameters == d
        va_mask = val_diameters == d
        te_mask = test_diameters == d
        tr_n = tr_mask.sum()
        va_n = va_mask.sum()
        te_n = te_mask.sum()
        tr_mean = np.mean(train_losses[tr_mask]) if tr_n > 0 else float("nan")
        tr_med = np.median(train_losses[tr_mask]) if tr_n > 0 else float("nan")
        va_mean = np.mean(val_losses[va_mask]) if va_n > 0 else float("nan")
        va_med = np.median(val_losses[va_mask]) if va_n > 0 else float("nan")
        te_mean = np.mean(test_losses[te_mask]) if te_n > 0 else float("nan")
        te_med = np.median(test_losses[te_mask]) if te_n > 0 else float("nan")

        if has_ap:
            tr_ap = _safe_ap(train_y[tr_mask], train_p[tr_mask]) if tr_n > 0 else float("nan")
            va_ap = _safe_ap(val_y[va_mask], val_p[va_mask]) if va_n > 0 else float("nan")
            te_ap = _safe_ap(test_y[te_mask], test_p[te_mask]) if te_n > 0 else float("nan")
            print(f"  {d:5d} | {tr_n:8d} {tr_mean:9.5f} {tr_med:8.5f} {tr_ap:8.4f} "
                  f"| {va_n:6d} {va_mean:9.5f} {va_med:8.5f} {va_ap:8.4f} "
                  f"| {te_n:7d} {te_mean:9.5f} {te_med:8.5f} {te_ap:8.4f}", flush=True)
        else:
            print(f"  {d:5d} | {tr_n:8d} {tr_mean:11.5f} {tr_med:10.5f} "
                  f"| {va_n:6d} {va_mean:9.5f} {va_med:8.5f} "
                  f"| {te_n:7d} {te_mean:10.5f} {te_med:9.5f}", flush=True)

    # --- Box plot (independent y-axes) ---
    split_data = [
        (train_diameters, train_losses, "Train", "#4C72B0"),
        (val_diameters, val_losses, "Val", "#55A868"),
        (test_diameters, test_losses, "Test", "#DD8452"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False)

    for ax, (diameters, losses, label, color) in zip(axes, split_data):
        unique_d = sorted(np.unique(diameters))
        grouped = [losses[diameters == d] for d in unique_d]
        counts = [len(g) for g in grouped]

        bp = ax.boxplot(
            grouped,
            positions=range(len(unique_d)),
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(len(unique_d)))
        ax.set_xticklabels(
            [f"{d}\n(n={c})" for d, c in zip(unique_d, counts)],
            fontsize=7, rotation=45, ha="right",
        )
        ax.set_xlabel("Graph Diameter", fontsize=11)
        ax.set_ylabel("Per-sample Loss", fontsize=11)
        ax.set_title(f"{label} Set — {dataset_name}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Per-sample Loss Distribution by Graph Diameter — {dataset_name}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"error_by_diameter_{dataset_name}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to: {plot_path}", flush=True)

    # --- Also save a violin plot for denser view (independent y-axes) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7), sharey=False)

    for ax, (diameters, losses, label, color) in zip(axes2, split_data):
        unique_d = sorted(np.unique(diameters))
        grouped = [losses[diameters == d] for d in unique_d]
        counts = [len(g) for g in grouped]

        # Only include groups with >= 2 points for violin (needs variance)
        valid_idx = [i for i, g in enumerate(grouped) if len(g) >= 2]
        if valid_idx:
            valid_grouped = [grouped[i] for i in valid_idx]
            valid_positions = list(range(len(valid_idx)))
            valid_unique_d = [unique_d[i] for i in valid_idx]
            valid_counts = [counts[i] for i in valid_idx]

            vp = ax.violinplot(
                valid_grouped,
                positions=valid_positions,
                showmedians=True,
                showextrema=True,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.6)

            ax.set_xticks(valid_positions)
            ax.set_xticklabels(
                [f"{d}\n(n={c})" for d, c in zip(valid_unique_d, valid_counts)],
                fontsize=7, rotation=45, ha="right",
            )
        ax.set_xlabel("Graph Diameter", fontsize=11)
        ax.set_ylabel("Per-sample Loss", fontsize=11)
        ax.set_title(f"{label} Set — {dataset_name}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

    fig2.suptitle(
        f"Per-sample Loss Distribution by Graph Diameter (Violin) — {dataset_name}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plot_path2 = os.path.join(save_dir, f"error_by_diameter_violin_{dataset_name}.png")
    fig2.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Violin plot saved to: {plot_path2}", flush=True)


if __name__ == "__main__":
    main()

