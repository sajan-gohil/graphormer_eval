import math
from typing import Iterable, List, Sequence, Tuple

import torch


RECURRENT_PARAM_NAMES = ("B_re", "B_im", "nu_log", "theta_log", "gamma_log")
NO_DECAY_KEYWORDS = ("embedding", "bias", "scale", "norm")


def split_parameter_groups(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    recurrent, no_decay, regular = [], [], []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        if any(key in name for key in RECURRENT_PARAM_NAMES):
            recurrent.append(param)
            continue

        lname = name.lower()
        if any(key in lname for key in NO_DECAY_KEYWORDS):
            no_decay.append(param)
        else:
            regular.append(param)

    return recurrent, no_decay, regular


def warmup_cosine_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if total_steps <= 1:
        return 1.0

    if step < warmup_steps:
        warmup_progress = float(step + 1) / float(max(warmup_steps, 1))
        return min_ratio + (1.0 - min_ratio) * warmup_progress

    decay_progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps - 1, 1))
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_grouped_optimizer_and_scheduler(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    lr_max: float,
    lr_min: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float = 0.05,
    recurrent_lr_factor: float = 1.0,
):
    recurrent, no_decay, regular = split_parameter_groups(named_parameters)

    param_groups = []
    if recurrent:
        param_groups.append({
            "params": recurrent,
            "lr": lr_max * recurrent_lr_factor,
            "weight_decay": 0.0,
        })
    if no_decay:
        param_groups.append({
            "params": no_decay,
            "lr": lr_max,
            "weight_decay": 0.0,
        })
    if regular:
        param_groups.append({
            "params": regular,
            "lr": lr_max,
            "weight_decay": weight_decay,
        })

    if not param_groups:
        raise ValueError("No trainable parameters available for optimizer construction")

    optimizer = torch.optim.AdamW(param_groups)

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        lr_min=lr_min,
        warmup_ratio=warmup_ratio,
    )
    return optimizer, scheduler


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    lr_min: float,
    warmup_ratio: float = 0.05,
):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        min_ratio = min(1.0, max(0.0, lr_min / max(base_lr, 1e-12)))

        lr_lambdas.append(
            lambda step, min_ratio=min_ratio: warmup_cosine_lambda(
                step=step,
                total_steps=max(total_steps, 1),
                warmup_steps=warmup_steps,
                min_ratio=min_ratio,
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
