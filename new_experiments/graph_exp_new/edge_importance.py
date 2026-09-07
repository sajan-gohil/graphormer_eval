"""
edge_importance.py
Edge-feature permutation-importance on a pretrained checkpoint (test #1 only).

Loads a saved model, runs three eval passes on a split and reports the metric:
    baseline        — unperturbed
    edge-permuted   — edge_attr rows shuffled across the batch (edge STRUCTURE
                      intact via edge_index; only the edge FEATURES are broken)
    node-permuted   — node feature rows shuffled (destroys node identity)

Interpretation:
    drop_edge = baseline - metric(edge-permuted)
    drop_node = baseline - metric(node-permuted)
  drop_edge ≈ 0  while  drop_node ≫ 0   ⇒  the model barely uses edge features
  (i.e. "good node/graph features, weak edge features" — the hypothesis).

Also prints edge_attr shape/dtype so the true edge-feature dimension is known
(useful for setting `edge_feat_dim` in data.py).

Usage:
    python edge_importance.py --checkpoint checkpoints_hop_masked/best_PascalVOC-SP.pt \
        --split val --repeats 3
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from data import get_loaders
from metrics import build_task
from train_hop_masked_transformer_final import _move_batch_to_device
from model_hop_masked_transformer_final import HopMaskedTransformerModel

try:
    from model_hop_masked_optimized import OptimizedHopMaskedTransformerModel
    _HAVE_OPT = True
except Exception:
    _HAVE_OPT = False


def _build_model(a: dict, task, dataset_info):
    """Reconstruct the model from a saved args dict (mirrors the trainers)."""
    g = a.get
    common = dict(
        hidden_dim=a["hidden_dim"], num_heads=a["num_heads"], ffn_ratio=a["ffn_ratio"],
        num_layers=a["num_layers"], dropout=a["dropout"], max_hops=a["max_hops"],
        hop_mode=a["hop_mode"], hop_window=a["hop_window"], hop_file=g("hop_file"),
        num_global_heads=a["num_global_heads"], output_dim=task.output_dim,
        graph_pool=a["graph_pool"], task_level=task.level, dataset_name=dataset_info["name"],
        node_feat_dim=dataset_info.get("node_feat_dim"),
        lap_pe_dim=a["lap_pe_dim"] if a.get("use_lap_pe") else 0,
        block_diag_out=a["block_diag_out"], dynamic_cross_hop=a["dynamic_cross_hop"],
        norm_type=a["norm_type"], v_head_dim=g("v_head_dim"), mask_type=a["mask_type"],
        adj_self_loops=a["adj_self_loops"], use_moe_gating=a["use_moe_gating"],
        top_k=a["top_k"], gate_noise=a["gate_noise"], balance_coeff=a["balance_coeff"],
        entropy_coeff=a["entropy_coeff"], use_virtual_node=a["use_virtual_node"],
        num_post_gat_layers=a["num_post_gat_layers"], num_gat_heads=a["num_gat_heads"],
        cross_hop_hop_embedding=a["cross_hop_hop_embedding"],
        use_edge_features=a["use_edge_features"], edge_feat_dim=dataset_info.get("edge_feat_dim"),
        cross_hop_no_ffn=a["cross_hop_no_ffn"], blend_adj_power=a["blend_adj_power"],
        use_edge_bias=a["use_edge_bias"], use_rrwp=a["use_rrwp"], rrwp_dim=a["rrwp_dim"],
        multihop_attn=a["multihop_attn"], multihop_readout=a["multihop_readout"],
        multihop_include_global=not a["multihop_no_global"], embed_dropout=a["dropout"],
    )
    if _HAVE_OPT and "use_normalized_head" in a:
        return OptimizedHopMaskedTransformerModel(
            **common,
            use_normalized_head=a.get("use_normalized_head", False),
            normalized_head_temp=a.get("normalized_head_temp", 10.0),
            use_mup_init=a.get("use_mup_init", False),
        )
    return HopMaskedTransformerModel(**common)


@torch.no_grad()
def _evaluate(model, loader, task, device, perturb: str, seed: int = 0):
    """perturb in {'none','edge','node'}. Returns the task metric."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    preds, labels = [], []
    for batch in loader:
        pyg, dm, nm = _move_batch_to_device(batch, device)
        if perturb == "edge":
            ea = getattr(pyg, "edge_attr", None)
            if ea is not None and ea.numel() > 0:
                perm = torch.randperm(ea.size(0), generator=g).to(device)
                pyg.edge_attr = ea[perm]
        elif perturb == "node":
            perm = torch.randperm(pyg.x.size(0), generator=g).to(device)
            pyg.x = pyg.x[perm]
        logits, *_ = model(pyg, dm, nm)
        preds.append(task.predict(logits))
        labels.append(task.labels_to_numpy(pyg.y))
    return task.compute_metric(np.concatenate(preds, 0), np.concatenate(labels, 0))


def main():
    p = argparse.ArgumentParser(description="Edge-feature permutation importance")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--repeats", type=int, default=3,
                   help="Averaging repeats for the (random) permutations.")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt["args"]
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')} "
          f"(saved best_val={ckpt.get('best_val','?')})", flush=True)

    train_loader, val_loader, test_loader, _, _, _, dataset_info = get_loaders(
        batch_size=a["batch_size"], num_workers=0, use_dist_masks=True,
        max_hops=a["max_hops"], dist_mask_workers=a.get("dist_mask_workers", 8),
        use_lap_pe=a.get("use_lap_pe", False), lap_pe_dim=a.get("lap_pe_dim", 8),
        dataset_name=a["dataset"], return_info=True,
        subgraph_mode=a.get("subgraph_mode", "partition"), num_parts=a.get("num_parts", 128),
        egonet_hops=a.get("egonet_hops", 2), egonet_max_nodes=a.get("egonet_max_nodes", 1024),
        max_egonet_samples=a.get("max_egonet_samples"), seed=a.get("seed", 0),
    )
    loader = val_loader if args.split == "val" else test_loader

    task = build_task(dataset_info["name"], dataset_info=dataset_info)
    task.loss_fn = task.loss_fn.to(device)

    model = _build_model(a, task, dataset_info).to(device)
    model.load_state_dict(ckpt["model"])

    # Report the true edge-feature dimension from the first batch.
    b0 = next(iter(loader))[0]
    ea = getattr(b0, "edge_attr", None)
    if ea is None or ea.numel() == 0:
        print("edge_attr: NONE (dataset has no edge features; edge test is a no-op)", flush=True)
    else:
        print(f"edge_attr: shape={tuple(ea.shape)} dtype={ea.dtype} "
              f"sample={ea.reshape(-1, ea.shape[-1] if ea.dim() > 1 else 1)[:3].tolist()}",
              flush=True)

    metric_name = getattr(task, "metric_name", "metric")
    base = _evaluate(model, loader, task, device, "none")
    edge_vals = [_evaluate(model, loader, task, device, "edge", s) for s in range(args.repeats)]
    node_vals = [_evaluate(model, loader, task, device, "node", s) for s in range(args.repeats)]
    edge_m, node_m = float(np.mean(edge_vals)), float(np.mean(node_vals))

    print("\n=== Permutation importance ({} split, {}) ===".format(args.split, metric_name))
    print(f"  baseline              : {base:.4f}")
    print(f"  edge-permuted         : {edge_m:.4f}   drop_edge = {base - edge_m:+.4f}")
    print(f"  node-permuted         : {node_m:.4f}   drop_node = {base - node_m:+.4f}")
    denom = (base - node_m) if abs(base - node_m) > 1e-9 else float("nan")
    print(f"  edge/node reliance    : {(base - edge_m) / denom:.3f}  "
          f"(≈0 ⇒ edges barely used; ~1 ⇒ edges as important as nodes)")


if __name__ == "__main__":
    main()

