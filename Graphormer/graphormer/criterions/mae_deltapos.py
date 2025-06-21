# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Mapping, Sequence, Tuple
from numpy import mod
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class IS2RECriterion(nn.Module):
    e_thresh = 0.02
    e_mean = -1.4729953244844094
    e_std = 2.2707848125378405
    d_mean = [0.1353900283575058, 0.06877671927213669, 0.08111362904310226]
    d_std = [1.7862379550933838, 1.78688645362854, 0.8023099899291992]

    def __init__(
        self,
        node_loss_weight=1.0,
        min_node_loss_weight=0.0,
        max_update=100000,
    ):
        super().__init__()
        self.node_loss_weight = node_loss_weight
        self.min_node_loss_weight = min_node_loss_weight
        self.max_update = max_update
        self.node_loss_weight_range = max(
            0, self.node_loss_weight - self.min_node_loss_weight
        )

    def forward(
        self,
        model: Callable[..., Tuple[Tensor, Tensor, Tensor]],
        sample: Mapping[str, Mapping[str, Tensor]],
        num_updates: int,
        reduce=True,
    ):
        assert num_updates >= 0
        node_loss_weight = (
            self.node_loss_weight
            - self.node_loss_weight_range * num_updates / self.max_update
        )

        output, node_output, node_target_mask = model(
            **sample["net_input"],
        )

        relaxed_energy = sample["targets"]["relaxed_energy"]
        relaxed_energy = relaxed_energy.float()
        relaxed_energy = (relaxed_energy - self.e_mean) / self.e_std
        loss = F.l1_loss(output.float().view(-1), relaxed_energy, reduction="sum")

        deltapos = sample["targets"]["deltapos"].float()
        deltapos = (deltapos - deltapos.new_tensor(self.d_mean)) / deltapos.new_tensor(
            self.d_std
        )
        deltapos *= node_target_mask
        node_output *= node_target_mask
        target_cnt = node_target_mask.sum(dim=[1, 2])
        node_loss = (
            F.l1_loss(node_output.float(), deltapos, reduction="none")
            .mean(dim=-1)
            .sum(dim=-1)
            / target_cnt
        ).sum()

        return loss + node_loss_weight * node_loss
