"""
Trainer for the OPTIMIZED hop-masked transformer.

Reuses the base trainer's data loading, task construction, evaluation epoch,
per-head diagnostics and error-by-diameter analysis (imported from
``train_hop_masked_transformer_final``), and swaps in:

    * OptimizedHopMaskedTransformerModel  (normalized head + muP init options;
      the 3 attention-level masking improvements are already in the base model)
    * SAM optimizer wrapper                (two-pass sharpness-aware step)
    * EMA / SWA-style iterate averaging     (eval on averaged weights)
    * lazy / two-timescale phase            (freeze backbone -> convex head)
    * SGLD Langevin annealing               (asymptotic global-min refinement)

Every base flag (--dataset, --hidden_dim, --hop_mode, --num_heads, ...) still
works.  New flags are grouped under "optimized" below.  See the module footer /
chat for command examples.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from data import get_loaders
from metrics import build_task
from model_hop_masked_transformer_final import set_attn_diagnostics
from model_hop_masked_optimized import (
    OptimizedHopMaskedTransformerModel,
    EMA,
    langevin_noise_,
    build_optimized_optimizer_and_scheduler,
)
from train_hop_masked_transformer_final import (
    build_parser,
    run_epoch,               # used for eval passes (optimizer=None)
    _run_link_epoch,         # link-level train/eval (SAM/EMA/SGLD aware)
    build_pos_or_class_weight,
    _log_head_stats,
    _move_batch_to_device,
    _plot_error_by_diameter,
)


# ---------------------------------------------------------------------------
# Argument parsing: base parser + optimized group.
# ---------------------------------------------------------------------------
def build_optimized_parser():
    p = build_parser()
    g = p.add_argument_group("optimized")

    # -- Architecture (raise the PL constant) --
    g.add_argument("--use_normalized_head", action="store_true", default=False,
                   help="Cosine logits + learnable temperature -> bounded logits "
                        "-> lower-bounds outer CE curvature (μ_outer>0). "
                        "Classification tasks only.")
    g.add_argument("--normalized_head_temp", type=float, default=10.0,
                   help="Initial softmax temperature τ for the normalized head.")
    g.add_argument("--use_mup_init", action="store_true", default=False,
                   help="muP-style width-stable init (Linear std=1/sqrt(fan_in)) "
                        "to keep the tangent kernel well-conditioned.")

    # -- Lazy / two-timescale --
    g.add_argument("--lazy_start_epoch", type=int, default=-1,
                   help="Freeze the backbone from this epoch on (only the head "
                        "trains -> convex CE, exact PL). -1 = disabled.")

    # -- SAM --
    g.add_argument("--use_sam", action="store_true", default=False,
                   help="Sharpness-Aware Minimization (two forward/backward "
                        "passes per step) -> flat, wide basins.")
    g.add_argument("--sam_rho", type=float, default=0.05,
                   help="SAM neighborhood radius ρ.")
    g.add_argument("--sam_adaptive", action="store_true", default=False,
                   help="Use ASAM (scale-invariant radius).")

    # -- EMA / SWA --
    g.add_argument("--use_ema", action="store_true", default=False,
                   help="Track an EMA of weights and evaluate/save on it.")
    g.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay.")

    # -- SGLD (Langevin annealing) --
    g.add_argument("--use_sgld", action="store_true", default=False,
                   help="Inject annealed Langevin noise after each step "
                        "(asymptotic global guarantee; use as a refinement).")
    g.add_argument("--sgld_start_epoch", type=int, default=0,
                   help="Only inject SGLD noise from this epoch onward "
                        "(0 = from the start).")
    g.add_argument("--sgld_beta0", type=float, default=1e4,
                   help="Initial inverse temperature β (T=1/β). Larger = less "
                        "noise. Annealed up by --sgld_beta_growth per step.")
    g.add_argument("--sgld_beta_growth", type=float, default=1.0,
                   help="Per-step multiplier on β (>1 anneals T -> 0). "
                        "1.0 = constant temperature.")
    return p


def parse_args():
    args, unknown = build_optimized_parser().parse_known_args()
    print(args.__dict__)
    print("Unknown args = ", unknown)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ---------------------------------------------------------------------------
# Optimized training epoch (supports SAM two-pass, SGLD, EMA).
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, task, device, optimizer, scheduler,
                    grad_clip, use_sam, ema, sgld):
    """One training epoch. ``sgld`` is a mutable dict {'beta','growth'} or None."""
    # Link-level tasks: reuse the shared pair-scoring loop (SAM/EMA/SGLD aware).
    if getattr(task, "level", None) == "link":
        loss, metric, _ = _run_link_epoch(
            model, loader, task, device, optimizer=optimizer, scheduler=scheduler,
            grad_clip=grad_clip, sam=use_sam, ema=ema, sgld=sgld)
        return loss, metric

    model.train()
    losses, preds_acc, labels_acc = [], [], []

    for batch in loader:
        pyg_batch, dist_masks, node_masks = _move_batch_to_device(batch, device)

        def _forward():
            logits, _, aux, _ = model(pyg_batch, dist_masks, node_masks)
            task_loss = task.loss(logits, pyg_batch.y)
            return task_loss + aux, task_loss, logits

        optimizer.zero_grad()
        loss, task_loss, logits = _forward()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        if use_sam:
            def closure():
                optimizer.zero_grad()
                l, _, _ = _forward()
                l.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                return l
            optimizer.step(closure)          # ascend to θ+δ, re-eval, descend
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if sgld is not None:
            lr_now = optimizer.param_groups[0]["lr"]
            langevin_noise_(model.parameters(), lr=lr_now, beta=sgld["beta"])
            sgld["beta"] *= sgld["growth"]

        if ema is not None:
            ema.update(model)

        losses.append(task_loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(pyg_batch.y))

    y_pred = np.concatenate(preds_acc, axis=0)
    y_true = np.concatenate(labels_acc, axis=0)
    metric = task.compute_metric(y_pred, y_true)
    return float(np.mean(losses)), float(metric)


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

    # multi_label -> BCE pos_weight; multiclass -> inverse-frequency CE weights.
    pos_weight = build_pos_or_class_weight(args, dataset_info, train_loader, args.device)

    task = build_task(dataset_name, dataset_info=dataset_info, pos_weight=pos_weight,
                      focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    task.loss_fn = task.loss_fn.to(args.device)

    if args.use_normalized_head and (task.level == "link"
                                     or task.task_type not in ("multiclass", "multi_label")):
        print(f"[warn] --use_normalized_head is for classification; "
              f"level={task.level} task_type={task.task_type}. Disabling it.", flush=True)
        args.use_normalized_head = False

    # ---- Optimized model ----
    model = OptimizedHopMaskedTransformerModel(
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
        embed_dropout=args.dropout,
        # optimized-only:
        use_normalized_head=args.use_normalized_head,
        normalized_head_temp=args.normalized_head_temp,
        use_mup_init=args.use_mup_init,
    ).to(args.device)

    print(f"[optimized] normalized_head={args.use_normalized_head} "
          f"mup_init={args.use_mup_init} sam={args.use_sam}(ρ={args.sam_rho}) "
          f"ema={args.use_ema}(d={args.ema_decay}) sgld={args.use_sgld} "
          f"lazy_start={args.lazy_start_epoch}", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.3f}M", flush=True)

    # ---- Optimizer + schedulers ----
    total_steps = max(args.max_epochs * len(train_loader), 1)
    optimizer, scheduler = build_optimized_optimizer_and_scheduler(
        model=model,
        lr_max=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        sam_adaptive=args.sam_adaptive,
    )
    plateau_mode = "max" if task.higher_is_better else "min"
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=plateau_mode, factor=0.5,
        patience=args.reduce_lr_patience, min_lr=args.lr_min,
    )

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    sgld = {"beta": args.sgld_beta0, "growth": args.sgld_beta_growth} if args.use_sgld else None

    best_val = -float("inf") if task.higher_is_better else float("inf")
    best_test, best_epoch, epochs_since_improve = None, -1, 0

    _log_interval = args.log_head_stats_interval
    _want_head_stats = _log_interval > 0
    _lazy_done = False

    for epoch in range(args.max_epochs):
        t0 = time.time()

        # -- lazy / two-timescale switch --
        if args.lazy_start_epoch >= 0 and epoch >= args.lazy_start_epoch and not _lazy_done:
            model.set_backbone_trainable(False)
            _lazy_done = True
            print(f"[lazy] epoch {epoch}: backbone frozen; training head only "
                  f"(convex CE, exact PL).", flush=True)

        # -- only anneal SGLD after its start epoch --
        sgld_active = sgld if (args.use_sgld and epoch >= args.sgld_start_epoch) else None

        tr_loss, tr_metric = train_one_epoch(
            model, train_loader, task, args.device, optimizer, scheduler,
            grad_clip=args.grad_clip, use_sam=args.use_sam, ema=ema, sgld=sgld_active,
        )

        # -- evaluate on EMA weights if enabled --
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)

        _collect_gw = _want_head_stats and (epoch % _log_interval == 0)
        if _collect_gw:
            set_attn_diagnostics(True)
        va_loss, va_metric, _gw_mean = run_epoch(
            model, val_loader, task, args.device, collect_gate_weights=_collect_gw)
        set_attn_diagnostics(False)
        te_loss, te_metric, _ = run_epoch(model, test_loader, task, args.device)

        plateau_scheduler.step(va_metric)
        dt = time.time() - t0

        improved = (va_metric > best_val if task.higher_is_better
                    else va_metric < best_val)
        if improved:
            best_val, best_test, best_epoch = va_metric, te_metric, epoch
            epochs_since_improve = 0
            # NOTE: when EMA is on, current weights ARE the EMA weights here.
            opt_state = (optimizer.base_optimizer if args.use_sam
                         else optimizer).state_dict()
            torch.save(
                {"model": model.state_dict(), "optimizer": opt_state,
                 "args": vars(args), "epoch": epoch, "best_val": best_val,
                 "best_test": best_test, "best_epoch": best_epoch},
                os.path.join(args.save_dir, f"best_{dataset_name}.pt"),
            )
        else:
            epochs_since_improve += 1

        # -- restore live (non-EMA) weights for the next training epoch --
        if ema is not None:
            ema.restore(model)

        ml = task.metric_label
        print(f"epoch {epoch:03d} | {dt:5.1f}s | "
              f"train loss {tr_loss:.4f} {ml} {tr_metric:.4f} | "
              f"val loss {va_loss:.4f} {ml} {va_metric:.4f} | "
              f"test loss {te_loss:.4f} {ml} {te_metric:.4f} | "
              f"best val {best_val:.4f} (epoch {best_epoch}, test {best_test})",
              flush=True)

        if _collect_gw:
            _log_head_stats(epoch, model, _gw_mean, args)

        if epochs_since_improve >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    print(f"BEST: val {ml} {best_val:.4f} | test {ml} {best_test:.4f} "
          f"(epoch {best_epoch})", flush=True)

    # -- error-by-diameter on the best checkpoint --
    ckpt_path = os.path.join(args.save_dir, f"best_{dataset_name}.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device)["model"])
    _plot_error_by_diameter(model, train_loader, val_loader, test_loader,
                            task, args.device, dataset_name, args.save_dir)


if __name__ == "__main__":
    main()

