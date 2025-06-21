# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass, field

import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os
import time

from ..base import BaseCriterion, BaseConfig

@dataclass
class DiffusionLossConfig(BaseConfig):
    valid_times: int = field(
        default=1, metadata={"help": "number of times to run validation"}
    )


class DiffusionLoss(BaseCriterion):
    def __init__(self, valid_times=1):
        super().__init__()
        self.valid_times = valid_times

    def forward(self, model, sample, reduce=True):
        if model.training:
            output = model.get_training_output(**sample["net_input"])
        else:
            with torch.no_grad():
                output = model.get_sampling_output(
                    **sample["net_input"], sampling_times=self.valid_times
                )

        persample_loss = output["persample_loss"]
        loss = torch.sum(persample_loss)
        return loss

