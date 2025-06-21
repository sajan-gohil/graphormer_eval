# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


class GraphPredictionL1Loss(nn.Module):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])

        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])
        return loss


class GraphPredictionL1LossWithFlag(GraphPredictionL1Loss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, perturb, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])
        return loss
