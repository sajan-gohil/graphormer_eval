"""
Optimized hop-masked transformer + numerical-optimization toolkit.

This module layers the convergence-oriented improvements discussed on top of
``model_hop_masked_transformer_final.py``.  The base model already contains the
three attention-level changes:

    (1) diameter-exceeding heads promoted to global,
    (2) random global-head promotion with prob ``dropout/2`` (train only),
    (3) masked attention dropout (dropout budget spent only on allowed pairs).

On top of those, this file adds the optimization-theory levers, split into two
groups by *what they act on*:

    A. ARCHITECTURE (raise the PL constant / condition the tangent kernel)
       - NormalizedHead .......... bounds logits  -> lower-bounds ∇²_z CE  (μ_outer > 0)
       - mup_init_ ............... width-stable init -> keeps λ_min(J Jᵀ) > 0
       - set_backbone_trainable .. lazy / two-timescale phase; freezing the
                                   backbone makes CE over the linear head convex
                                   (exact PL on the head).

    B. NUMERICAL OPTIMIZATION (act on the iterates / gradients)
       - SAM ..................... sharpness-aware minimization -> flat, wide basins
       - langevin_noise_ ........ SGLD annealing  -> asymptotic *global* guarantee
       - perturbed_kick_ ........ perturbed SGD   -> escape strict saddles (local min)
       - EMA .................... iterate averaging (SWA-style) -> basin centre
       - pl_ratio ............... empirical PL / gradient-dominance diagnostic

None of these can *hard-guarantee* a global minimum for a non-convex net.  The
honest hierarchy they implement:
    Tier 0 (arch, A): make PL hold with a usable μ  -> linear rate to global VALUE.
    Tier 1 (Langevin): asymptotic global guarantee (expensive).
    Tier 2 (perturbed / SAM): guaranteed 2nd-order stationary point (local min).
    Tier 3 (EMA, preconditioning): faster / better empirical minima.

The model class is a thin subclass of the base model, so it keeps the exact
same ``forward(batch, dist_masks, node_masks, return_gate_weights=False)``
signature and return tuple ``(logits, node_emb, aux_loss, gate_weights)``.
Everything is drop-in with ``train_hop_masked_transformer_final.py``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_hop_masked_transformer_final import HopMaskedTransformerModel


# ===========================================================================
# A. ARCHITECTURE IMPROVEMENTS
# ===========================================================================
class NormalizedHead(nn.Module):
    """Cosine-similarity classification head with a learnable temperature.

    Standard CE-softmax is strongly convex in the logits *only on a bounded
    set*: as ``‖z‖ -> ∞`` the outer Hessian ``∇²_z CE`` vanishes, so the outer
    curvature term in ``μ ≳ λ_min(∇²_z CE) · λ_min(J Jᵀ)`` collapses.  L2-
    normalising both the feature and the class weights caps ``‖z‖ ≤ τ``, which
    lower-bounds ``λ_min(∇²_z CE)`` away from 0 and keeps μ usable.

    Drop-in for the base model's ``self.head`` (same call: ``head(x) -> logits``).

    NOTE: this is a *classification* head (meant for cross-entropy). For
    regression targets (e.g. Peptides-struct) keep ``use_normalized_head=False``
    and use the base MLP head.
    """

    def __init__(self, hidden_dim: int, output_dim: int,
                 init_temp: float = 10.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim, output_dim, bias=False)
        # Parameterise temperature in log-space so it stays positive.
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.proj(self.norm(x))))
        x = F.normalize(x, dim=-1)                       # ‖feature‖ = 1
        w = F.normalize(self.cls.weight, dim=-1)          # ‖class row‖ = 1
        return self.log_temp.exp() * F.linear(x, w)       # logits in [-τ, τ]


@torch.no_grad()
def mup_init_(module: nn.Module) -> None:
    """muP-style re-initialisation: every Linear gets std = 1/sqrt(fan_in).

    Width-invariant activation/gradient scale keeps the empirical NTK (tangent
    kernel) from degenerating as the model is widened, so ``λ_min(J Jᵀ)`` — and
    hence the PL constant μ — stays bounded away from 0 in the over-parameterized
    regime where PL provably holds.  Apply *after* model construction:

        model = OptimizedHopMaskedTransformerModel(...)
        model.apply(mup_init_)   # or pass use_mup_init=True
    """
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        nn.init.normal_(module.weight, mean=0.0, std=(1.0 / fan_in) ** 0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class OptimizedHopMaskedTransformerModel(HopMaskedTransformerModel):
    """Base hop-masked transformer + PL-oriented architecture options.

    Extra kwargs (everything else is forwarded unchanged to the base model):
        use_normalized_head:   replace the MLP head with ``NormalizedHead``
                               (bounded logits -> μ_outer > 0). Classification only.
        normalized_head_temp:  initial softmax temperature τ for that head.
        use_mup_init:          apply width-stable muP init after construction.

    The three attention-level improvements (global promotion, random promotion,
    masked attention dropout) live in the base class and are always active.
    """

    def __init__(
        self,
        *args,
        use_normalized_head: bool = False,
        normalized_head_temp: float = 10.0,
        use_mup_init: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_normalized_head = use_normalized_head

        if use_normalized_head:
            # Recover the head widths from the base MLP head (last Linear -> out).
            out_dim = None
            for m in reversed(self.head):
                if isinstance(m, nn.Linear):
                    out_dim = m.out_features
                    break
            if out_dim is None:
                raise RuntimeError("Could not infer output_dim from base head.")
            # Reuse the same dropout rate the base model was built with.
            self.head = NormalizedHead(
                self.hidden_dim, out_dim,
                init_temp=normalized_head_temp,
                dropout=getattr(self, "dropout", 0.0),
            )

        if use_mup_init:
            self.apply(mup_init_)

    # -- Lazy / two-timescale training -------------------------------------
    def set_backbone_trainable(self, flag: bool) -> None:
        """Freeze (flag=False) / unfreeze (flag=True) everything except the head.

        With the backbone frozen, the features ``φ = Φ(X)`` are fixed and CE over
        the linear/normalized head is convex — strongly convex when the feature
        Gram ``φᵀφ ≻ 0`` — so PL holds *exactly* on the trained parameters and
        the head converges to its global optimum at a linear rate.  Typical use:
        train the full model for a warmup, then flip to lazy for a refinement
        phase.
        """
        for name, p in self.named_parameters():
            if not name.startswith("head"):
                p.requires_grad_(flag)


# ===========================================================================
# B. NUMERICAL OPTIMIZATION UTILITIES
# ===========================================================================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021).

    Minimises ``max_{‖δ‖≤ρ} F(θ+δ)`` instead of ``F(θ)``, biasing the optimizer
    toward *flat, wide* basins that empirically coincide with global-quality,
    well-generalising minima.  Wraps any base optimizer (e.g. AdamW) and needs
    two forward/backward passes per step.

    Training-loop usage (mirrors the base loop, adding a second closure pass;
    remember the model returns ``(logits, _, aux_loss, _)``):

        base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        optimizer = SAM(model.parameters(), base_optimizer=base, rho=0.05)

        def closure():
            optimizer.zero_grad()
            logits, _, aux, _ = model(pyg_batch, dist_masks, node_masks)
            loss = task.loss(logits, pyg_batch.y) + aux
            loss.backward()
            return loss

        # first pass computes the gradient used for the ascent step
        logits, _, aux, _ = model(pyg_batch, dist_masks, node_masks)
        loss = task.loss(logits, pyg_batch.y) + aux
        loss.backward()
        optimizer.step(closure)     # ascend to θ+δ, re-eval, descend
        scheduler.step()

    ``adaptive=True`` gives ASAM (scale-invariant radius).
    """

    def __init__(self, params, base_optimizer: torch.optim.Optimizer,
                 rho: float = 0.05, adaptive: bool = False):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        # Share param groups with the wrapped optimizer.
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group.setdefault("rho", rho)
            group.setdefault("adaptive", adaptive)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)                       # climb to the local worst-case
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])      # back to θ
        self.base_optimizer.step()                # descend with the sharp grad
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires a closure that re-evaluates the loss.")
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()                                 # recompute grad at θ+δ
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                .norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"] if p.grad is not None
            ]),
            p=2,
        )


@torch.no_grad()
def langevin_noise_(params: Iterable[torch.nn.Parameter],
                    lr: float, beta: float) -> None:
    """SGLD noise injection — call *after* ``optimizer.step()``.

    Turns SGD into Stochastic Gradient Langevin Dynamics:
        θ <- θ - lr·∇F + sqrt(2·lr/β)·ξ,   ξ ~ N(0, I)
    which samples the Gibbs measure ``∝ exp(-β F)``.  Annealing ``β -> ∞``
    (temperature ``T = 1/β -> 0``) concentrates the iterates on the *global*
    minimizers — the only Tier-1, asymptotically-global scheme here.  Use as a
    short low-temperature refinement phase; a full run is prohibitively slow.

        optimizer.step()
        langevin_noise_(model.parameters(), lr=cur_lr, beta=beta)
        beta *= beta_growth          # e.g. 1.02 per step, anneal T -> 0
    """
    std = (2.0 * lr / max(beta, 1e-12)) ** 0.5
    for p in params:
        if p.requires_grad:
            p.add_(torch.randn_like(p) * std)


@torch.no_grad()
def perturbed_kick_(params: Iterable[torch.nn.Parameter],
                    radius: float) -> None:
    """Perturbed-SGD saddle kick (Jin et al., 2017).

    Add one isotropic perturbation when the gradient is small (a plateau) to
    push the iterate off a strict saddle's stable manifold.  Under the strict-
    saddle property this yields convergence to a second-order stationary point
    (approximate local minimum) in polynomial time.

        if grad_norm < g_thresh and (t - t_last) > wait:
            perturbed_kick_(model.parameters(), radius=r)
            t_last = t
    """
    for p in params:
        if p.requires_grad:
            p.add_(torch.randn_like(p) * radius)


class EMA:
    """Exponential moving average of parameters (SWA-style iterate averaging).

    Averaging the tail of the trajectory lands in the *centre* of the (flat)
    basin rather than on its noisy rim, giving a lower effective loss and better
    generalisation.  Track during training, then evaluate with the averaged
    weights.

        ema = EMA(model, decay=0.999)
        ...
        optimizer.step(); ema.update(model)     # each step
        ...
        ema.store(model); ema.copy_to(model)    # before eval
        validate(model)
        ema.restore(model)                      # after eval, resume training
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k in self.shadow:
            msd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._backup:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}


def pl_ratio(loss: torch.Tensor,
             params: Sequence[torch.nn.Parameter],
             f_star_hat: float) -> float:
    """Empirical Polyak-Łojasiewicz / gradient-dominance ratio.

    Returns  ``‖∇F‖² / (2 (F - F̂*))``, a lower-bound estimate of the PL constant
    μ (using ``f_star_hat`` as a running proxy for F*).  If this stays bounded
    away from 0 across training, PL holds with that μ and the linear-rate bound
        E[F(θ_T)] - F*  ≤ (1-αμ)^T (F(θ_0)-F*) + Lασ²/(2μ)
    applies; if it collapses toward 0, widen the model or extend the lazy phase.
    Call sparingly (extra backward): it builds its own grad graph.
    """
    trainable = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
    g2 = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    denom = 2.0 * (loss.detach() - f_star_hat) + 1e-8
    return float(g2 / denom)


# ===========================================================================
# Convenience builder: AdamW (+ optional SAM) + warmup-cosine schedule
# ===========================================================================
def build_optimized_optimizer_and_scheduler(
    model: nn.Module,
    lr_max: float,
    lr_min: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float = 0.05,
    recurrent_lr_factor: float = 1.0,
    use_sam: bool = False,
    sam_rho: float = 0.05,
    sam_adaptive: bool = False,
):
    """AdamW with grouped weight decay (reusing ``optim_utils``) + warmup-cosine.

    When ``use_sam=True`` the AdamW is wrapped in :class:`SAM`; the scheduler is
    attached to the underlying AdamW so ``scheduler.step()`` works unchanged, and
    the training loop must call ``optimizer.step(closure)`` (see SAM docstring).
    """
    from optim_utils import (
        split_parameter_groups, build_warmup_cosine_scheduler,
    )

    recurrent, no_decay, regular = split_parameter_groups(model.named_parameters())
    param_groups = []
    if recurrent:
        param_groups.append({"params": recurrent,
                             "lr": lr_max * recurrent_lr_factor, "weight_decay": 0.0})
    if no_decay:
        param_groups.append({"params": no_decay, "lr": lr_max, "weight_decay": 0.0})
    if regular:
        param_groups.append({"params": regular, "lr": lr_max, "weight_decay": weight_decay})
    if not param_groups:
        raise ValueError("No trainable parameters available for optimizer construction")

    base = torch.optim.AdamW(param_groups)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=base, total_steps=total_steps, lr_min=lr_min, warmup_ratio=warmup_ratio,
    )
    optimizer = SAM(model.parameters(), base, rho=sam_rho, adaptive=sam_adaptive) if use_sam else base
    return optimizer, scheduler

