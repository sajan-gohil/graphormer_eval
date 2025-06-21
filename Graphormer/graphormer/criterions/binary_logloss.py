# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional


class GraphPredictionBinaryLogLoss(nn.Module):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])
        
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[: logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        )
        return loss


class GraphPredictionBinaryLogLossWithFlag(GraphPredictionBinaryLogLoss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[: logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        )
        return loss
