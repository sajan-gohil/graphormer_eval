# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional


class GraphPredictionMulticlassCrossEntropy(nn.Module):
    """
    Implementation for the multi-class log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])[: logits.size(0)]

        loss = functional.cross_entropy(
            logits, targets.reshape(-1), reduction="sum"
        )
        return loss


class GraphPredictionMulticlassCrossEntropyWithFlag(
    GraphPredictionMulticlassCrossEntropy
):
    """
    Implementation for the multi-class log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        perturb = sample.get("perturb", None)
        logits = model(**sample["net_input"], perturb=perturb)
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])[: logits.size(0)]

        loss = functional.cross_entropy(
            logits, targets.reshape(-1), reduction="sum"
        )
        return loss
