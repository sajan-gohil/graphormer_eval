from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

@dataclass
class BaseConfig:
    """Base configuration class"""
    pass

class BaseDataset:
    """Base dataset class"""
    def __init__(self):
        self.sizes = None
        self.supports_prefetch = False

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def size(self, index: int) -> int:
        """Return an example's size as a float or tensor."""
        if self.sizes is None:
            return 0
        return self.sizes[index]

    def num_tokens(self, index: int) -> int:
        return self.size(index)

    def num_tokens_vec(self, indices: List[int]) -> List[int]:
        return [self.num_tokens(i) for i in indices]

    def collater(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a list of samples to form a mini-batch."""
        raise NotImplementedError

class BaseEncoder(nn.Module):
    """Base encoder class"""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def max_positions(self) -> int:
        """Maximum length supported by the encoder."""
        return 1e6

class BaseModel(nn.Module):
    """Base model class"""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def max_positions(self) -> int:
        """Maximum length supported by the model."""
        return 1e6

class BaseCriterion(nn.Module):
    """Base criterion class"""
    def __init__(self):
        super().__init__()

    def forward(self, model, sample, reduce=True) -> Dict[str, Any]:
        raise NotImplementedError

class BaseTask:
    """Base task class"""
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        pass

    @classmethod
    def setup_task(cls, cfg: BaseConfig, **kwargs):
        """Setup the task."""
        return cls(cfg)

    def build_model(self, cfg: BaseConfig) -> BaseModel:
        """Build the model."""
        raise NotImplementedError

    def build_criterion(self, cfg: BaseConfig) -> BaseCriterion:
        """Build the criterion."""
        raise NotImplementedError

    def build_dataset(self, split: str, **kwargs) -> BaseDataset:
        """Build the dataset."""
        raise NotImplementedError

    def train_step(self, sample: Dict[str, Any], model: BaseModel, criterion: BaseCriterion, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Train step."""
        raise NotImplementedError

    def valid_step(self, sample: Dict[str, Any], model: BaseModel, criterion: BaseCriterion) -> Dict[str, Any]:
        """Validation step."""
        raise NotImplementedError 