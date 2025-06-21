import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"activation should be relu/gelu/tanh/linear, not {activation}")

def get_available_activation_fns() -> List[str]:
    return ["relu", "gelu", "tanh", "linear"]

def safe_getattr(obj: Any, k: str, default: Any = None) -> Any:
    """Safely get an attribute from an object."""
    return getattr(obj, k, default)

def safe_hasattr(obj: Any, k: str) -> bool:
    """Safely check if an object has an attribute."""
    return hasattr(obj, k)

def quant_noise(module: nn.Module, p: float, block_size: int):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to train, or how to quantize the resulting model
          to get the actual quantized model, please refer to the paper
    """
    if not p or p <= 0:
        return module

    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data = torch.quantize_per_tensor_dynamic(
            module.weight.data, qscheme=torch.per_tensor_affine, dtype=torch.qint8
        )
    elif isinstance(module, nn.Conv2d):
        module.weight.data = torch.quantize_per_tensor_dynamic(
            module.weight.data, qscheme=torch.per_tensor_affine, dtype=torch.qint8
        )
    else:
        raise NotImplementedError(f"Module {module.__class__.__name__} not supported for quantization")

    return module

class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this will iterate over layers [1, 3]
            x = layer(x)
        for layer in layers:  # this will iterate over layers [1, 2, 3]
            x = layer(x)
    """

    def __init__(self, p: float, modules: Optional[List[nn.Module]] = None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m

class LayerNorm(nn.Module):
    """Layer normalization module"""

    def __init__(self, embed_dim: int, export: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.export = export
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class FairseqDropout(nn.Module):
    """Dropout module"""

    def __init__(self, p: float, module_name: Optional[str] = None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x: Tensor, inplace: bool = False) -> Tensor:
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x 