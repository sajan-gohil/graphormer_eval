# coding=utf-8
# Copyright 2022 Microsoft, clefourrier The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Graphormer model."""
import logging
logging.basicConfig(level=logging.INFO)
import math
import wandb

from collections.abc import Iterable, Iterator
from typing import Optional, Union
import datetime

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss
from torch.optim import Adam
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_graphormer import GraphormerConfig

# Add import for diffusion
from graph_diffusion import GraphLatentDiffusion
import numpy as np

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "graphormer-base-pcqm4mv1"
_CONFIG_FOR_DOC = "GraphormerConfig"


def quant_noise(module: nn.Module, p: float, block_size: int):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # scale weights and apply mask
            mask = mask.to(torch.bool)  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


def compute_attention_snr(attn_weights, labels, node_mask):
    """
    Compute AttentionSNR: 10*log10(sum(attn_score_same_class)/sum(attn_score_diff_class))
    attn_weights: [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]
    labels: [batch, num_nodes] or [num_nodes]
    node_mask: optional mask for valid nodes
    """
    # if attn_weights is None or labels is None:
    #    return float('nan')
    if attn_weights.dim() == 4:
        # [num_heads, batch, num_nodes, num_nodes] -> mean over heads
        attn_weights = attn_weights.mean(dim=0)
    if attn_weights.dim() == 3:
        # [batch, num_nodes, num_nodes]
        batch_size = attn_weights.shape[0]
        snrs = []
        for i in range(batch_size):
            snrs.append(compute_attention_snr(attn_weights[i], labels[i], node_mask[i] if node_mask is not None else None))
        return float(torch.tensor(snrs).mean().item())
    # [num_nodes, num_nodes]
    num_nodes = attn_weights.shape[0]
    if node_mask is not None:
        valid = node_mask.bool()
        attn_weights = attn_weights[valid][:, valid]
        labels = labels[valid]
    same = (labels.unsqueeze(0) == labels.unsqueeze(1))
    diff = ~same
    attn_same = attn_weights[same].sum().item()
    attn_diff = attn_weights[diff].sum().item() + 1e-8
    if attn_same == 0 and attn_diff == 0:
        return float('nan')
    # print("SNR VALS = ", (attn_same/attn_diff), attn_same, attn_diff)
    snr = 10 * math.log10(attn_same / attn_diff) if attn_diff > 0 else float('inf')
    return snr


class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://huggingface.co/papers/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms

        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.feature_encoder = nn.Linear(1433, config.hidden_size)
        self.in_degree_encoder = nn.Embedding(
            config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.out_degree_encoder = nn.Embedding(
            config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.graph_token = nn.Embedding(1, config.hidden_size)  # Equivalent to [CLS]

    def forward(
        self,
        input_nodes: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
    ) -> torch.Tensor:
        n_graph, n_node = input_nodes.size()[:2]
        # print("Input nodes:", input_nodes.shape)
        if self.config.dataset_name not in ["pcqm4mv2"]:
            node_feature = (
               self.feature_encoder(input_nodes.to(dtype=torch.float32))  # nn.functional.relu(self.feature_encoder(input_nodes.to(dtype=torch.float32))))
               + self.in_degree_encoder(in_degree)
               + self.out_degree_encoder(out_degree)
            )
            # print("Processed")
        else:
            node_feature = (  # node feature + graph token
                self.atom_encoder(input_nodes).sum(dim=-2)  # [n_graph, n_node, n_hidden]
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
            )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # We do not change edge feature embedding learning, as edge embeddings are represented as a combination of the original features
        # + shortest path
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)

        self.edge_type = config.edge_type
        if self.edge_type == "multi_hop":
            # print("IN MULTIHOP SHOULD NOT BE HERE")
            self.edge_dis_encoder = nn.Embedding(
                config.num_edge_dis * config.num_attention_heads * config.num_attention_heads,
                1,
            )

        if config.enable_spatial_encoder:
            self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    def forward(
        self,
        input_nodes: torch.LongTensor,
        attn_bias: torch.Tensor,
        spatial_pos: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
    ) -> torch.Tensor:
        n_graph, n_node = input_nodes.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if self.config.enable_spatial_encoder:
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            # print("IN MULTIHOP 2 SHOULD NOT BE HERE")
            spatial_pos_ = spatial_pos.clone()

            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, input_nodes > 1 to input_nodes - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                try:
                    input_edges = input_edges[:, :, :, : self.multi_hop_max_dist, :]
                except:
                    print(type(input_edges), self.multi_hop_max_dist)
                    print(np.array(input_edges).shape)
                    raise
            # [n_graph, n_node, n_node, max_dist, n_head]

            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(
                1, 2, 3, 0, 4
            )
            input_edges = (input_edges.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim

        self.num_heads = config.num_attention_heads
        self.attention_dropout_module = torch.nn.Dropout(p=config.attention_dropout, inplace=False)

        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (self.head_dim * config.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        self.scaling = self.head_dim**-0.5

        self.self_attention = True  # config.self_attention
        if not (self.self_attention):
            raise NotImplementedError("The Graphormer model only supports self attention for now.")
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError("Self-attention requires query, key and value to be of the same size.")

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.q_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.out_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.onnx_trace = False
        if self.config.enable_layerwise_diffusion:
            self.diffusion_model = GraphLatentDiffusion(self.kdim, config.embedding_dim, config.diffusion_steps, config=self.config)

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def remove_attention_noise(self, attn_weights, q, k, v, batch_size, target_len, source_len, edge_index_list):
        # attn_weights = [bsz * self.num_heads, tgt_len, src_len]
        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        


    def forward(
        self,
        query: torch.LongTensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        edge_index_list: list = None,
        attn_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
            attn_override (torch.Tensor, optional): Override attention weights with this tensor.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len
        if not (embedding_dim == self.embedding_dim):
            raise AssertionError(
                f"The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim"
                f" {self.embedding_dim}."
            )
        if not (list(query.size()) == [tgt_len, bsz, embedding_dim]):
            raise AssertionError("Query size incorrect in Graphormer, compared to model dimensions.")

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if (key_bsz != bsz) or (value is None) or not (src_len, bsz == value.shape[:2]):
                    raise AssertionError(
                        "The batch shape does not match the key or value shapes provided to the attention."
                    )

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (k is None) or not (k.size(1) == src_len):
            raise AssertionError("The shape of the key generated in the attention is incorrect")

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len:
                raise AssertionError(
                    "The shape of the generated padding mask for the key does not match expected dimensions."
                )
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)  # Returns same thing

        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError("The attention weights generated do not match the expected dimensions.")

        if attn_bias is not None:  # centrality, edge, etc embeddings
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_override is not None:
            attn_weights = attn_override

        self.last_attn_logits = attn_weights

        if before_softmax:
            return attn_weights, v

        # --- Diffusion part ----
        if self.config.enable_layerwise_diffusion:
            attn_weights = self.remove_attention_noise(
                attn_weights, q, k, v, batch_size=bsz,
                target_len=tgt_len, source_len=src_len, edge_index_list=edge_index_list)
        # --- Diffusion part ----

        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)


        if v is None:
            raise AssertionError("No value generated")
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError("The attention generated do not match the expected dimensions.")

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn: torch.Tensor = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
        return attn_weights


class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(self, config: GraphormerConfig) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout, inplace=False)

        # Initialize blocks
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = GraphormerMultiheadAttention(config)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.fc1 = self.build_fc(
            self.embedding_dim,
            config.ffn_embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )
        self.fc2 = self.build_fc(
            config.ffn_embedding_dim,
            self.embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_fc(
        self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int
    ) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        input_nodes: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        attn_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            # need_weights=False,
            attn_mask=self_attn_mask,
            attn_override=attn_override,
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        return input_nodes, attn


class GraphormerGraphEncoder(nn.Module):
    def __init__(self, config: GraphormerConfig):
        super().__init__()

        self.config = config
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable

        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        if not self.config.remove_attn_bias:
            self.graph_attn_bias = GraphormerGraphAttnBias(config)

        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Apply initialization of model params after building the model
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.LongTensor] = None,
        attn_override: Optional[dict] = None,
    ) -> tuple[Union[torch.Tensor, list[torch.LongTensor]], torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        data_x = input_nodes
        # logging.info(f"DATA X SHAPE = , {tuple(data_x.shape)}")
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        if not self.config.remove_attn_bias:
            attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)

        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)

        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        input_nodes = self.dropout_module(input_nodes)

        input_nodes = input_nodes.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)

        for layer_idx, layer in enumerate(self.layers):
            # logging.info(f"Processing layer:, {layer}, FOR INPUT:, {tuple(input_nodes.shape)}")
            override = None
            if attn_override is not None and layer_idx in attn_override:
                override = attn_override[layer_idx]

            input_nodes, attn = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
                attn_override=override,
            )
            if not last_state_only:
                inner_states.append(input_nodes)

        graph_rep = input_nodes[0, :, :]

        if last_state_only:
            inner_states = [input_nodes]

        if self.traceable:
            return torch.stack(inner_states), graph_rep, attn  # last attn weight
        else:
            return inner_states, graph_rep, attn  # last attn weight


class GraphormerDecoderHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        """num_classes should be 1 for regression, or the number of classes for classification"""
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes


class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GraphormerConfig
    base_model_prefix = "graphormer"
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def normal_(self, data: torch.Tensor):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]):
        """
        Initialize the weights specific to the Graphormer Model.
        """
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, GraphormerMultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, GraphormerMultiheadAttention, GraphormerGraphEncoder
        ],
    ):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # We might be missing part of the Linear init, dependent on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GraphormerMultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphormerGraphEncoder):
            if module.apply_graphormer_init:
                module.apply(self.init_graphormer_params)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config: GraphormerConfig, enable_diffusion: bool = True):
        super().__init__(config)
        self.config = config
        self.max_nodes = config.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(config)

        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(config, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        if config.enable_diffusion:
            self.diffusion_model = GraphLatentDiffusion(
                input_dim=config.embedding_dim,
                latent_dim=config.embedding_dim,
                num_denoising_steps=config.diffusion_steps,
                config=config)
            # self.diffusion_optimizer = Adam(self.diffusion_model.parameters(), lr=1e-4)
        else:
            self.diffusion_model = None
            self.diffusion_optimizer = None

        self.post_init()

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
    
    def add_dummy_nodes(self, input_nodes, pad_size=100):
        # input_nodes: [batch, num_nodes, feature_dim]
        # input_edges: [batch, num_nodes, num_nodes, edge_feature_dim]
        batch_size, num_nodes, feature_dim = input_nodes.size()
        # add nodes as mean of randomly selected existing nodes
        node_indices = torch.randint(1, num_nodes, (pad_size,), device=input_nodes.device)
        node_padding = input_nodes[:, node_indices, :].mean(dim=1, keepdim=True).repeat(1, pad_size, 1)
        input_nodes = torch.cat([input_nodes, node_padding], dim=1)

        return input_nodes

    def calc_dummy_node_loss(self, input_nodes, attn_weight, num_original_nodes):
        batch_size, num_nodes, feature_dim = input_nodes.size()
        start_idx = num_original_nodes + 1
        if start_idx >= num_nodes:
            return 0.0
            
        # attn_weight: [batch, num_heads, num_nodes, num_nodes]
        # Rows corresponding to dummy nodes
        dummy_rows = attn_weight[:, :, start_idx:, :]
        # Cols corresponding to dummy nodes
        dummy_cols = attn_weight[:, :, :, start_idx:]
        dummy_overlap = attn_weight[:, :, start_idx:, start_idx:]
        
        # Sum of absolute values
        loss = dummy_rows.abs().sum() + dummy_cols.abs().sum() - 2*dummy_overlap.abs().sum()
        
        # Normalize
        num_dummy = num_nodes - start_idx
        loss = loss / (batch_size * attn_weight.size(1))
        loss = loss / (num_dummy * 2)
        loss = loss / num_nodes
        
        return loss 
        
    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb: Optional[torch.FloatTensor] = None,
        masked_tokens: None = None,
        return_dict: Optional[bool] = None,
        edge_index: Optional[torch.LongTensor] = None,
        log_step: Optional[int] = None,
        log_group: Optional[int] = None,
        attn_override: Optional[dict] = None,
#        **unused,
         **kwargs
    ) -> Union[tuple[torch.LongTensor], BaseModelOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.freeze_pretrained_encoder:
            with torch.no_grad():
                inner_states, graph_rep, attn_weight = self.graph_encoder(
                    input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb, edge_index=edge_index, attn_override=attn_override
                )
        else:
            # add dummy nodes here
            if self.config.node_augmentation and self.training:
                input_nodes = self.add_dummy_nodes(input_nodes, pad_size=self.config.aug_num_dummy_nodes)
                num_original_nodes = input_nodes.size(1)
            inner_states, graph_rep, attn_weight = self.graph_encoder(
                input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb, edge_index=edge_index, attn_override=attn_override
            )
            # Calculate loss based on attention weights for dummy nodes
            dummy_node_loss = torch.tensor(0.0, device=input_nodes.device)
            if self.config.node_augmentation and self.training:
                dummy_node_loss = self.calc_dummy_node_loss(
                    input_nodes, attn_weight, num_original_nodes=num_original_nodes
                )
                

        # --- AttentionSNR logging: before diffusion ---
        # Use last attn_weight (after softmax), and input_nodes before diffusion
        labels = kwargs.get('labels', None)
        node_mask = kwargs.get('node_mask', None)
        if attn_weight is not None and labels is not None:
            snr_attn = compute_attention_snr(attn_weight[:, 1:, 1:], labels, node_mask)
            if log_group and log_step:
                wandb.log({f"ASNR_{log_group}/attn_weight_before_diffusion": snr_attn,
                           "step": self.config.current_step})
        # Compute SNR from normalized dot product + softmax of input_nodes (before diffusion)
        # input_nodes: [batch, num_nodes+1, hidden_dim], remove graph token
        input_nodes_ = inner_states[-1].transpose(0, 1)[:, 1:, :]
        if labels is not None:
            # Compute dot product attention
            normed = input_nodes_ / (input_nodes_.norm(dim=-1, keepdim=True) + 1e-8)
            attn_sim = torch.matmul(normed, normed.transpose(1, 2))
            attn_sim = torch.softmax(attn_sim, dim=-1)
            snr_sim = compute_attention_snr(attn_sim, labels, node_mask)
            if log_group and log_step:
            #    print(f"Logging ASNR_{log_group}/dotprod_softmax_before_diffusion: {snr_sim}, current_step={self.config.current_step}")
                wandb.log({f"ASNR_{log_group}/dotprod_softmax_before_diffusion": snr_sim,
                           "step": self.config.current_step})

        # last inner state, then revert Batch and Graph len
        input_nodes = inner_states[-1].transpose(0, 1)

        # --- Diffusion integration ---
        if self.config.enable_diffusion:
            # input_nodes: [batch, num_nodes+1, hidden_dim] (includes graph token)
            # Remove graph token for diffusion, then re-attach after
            graph_token = input_nodes[:, :1, :]
            node_emb = input_nodes[:, 1:, :]
            # edge_index should be a list of edge_index tensors for each graph in batch
            node_emb, attention_matching_loss = self.diffusion_model(
                node_emb, edge_index,
                aug_added_edges=kwargs.get("aug_added_edges", None),
                aug_removed_edges=kwargs.get("aug_removed_edges", None),
                aug_original_edges=kwargs.get("aug_original_edges", None),
                labels=labels
            )
            input_nodes = torch.cat([graph_token, node_emb], dim=1)

            # --- AttentionSNR logging: after diffusion ---
            labels = kwargs.get('labels', None)
            node_mask = kwargs.get('node_mask', None)
            if labels is not None:
                normed = node_emb / (torch.sqrt(torch.tensor(node_emb.shape[-1])) + 1e-8)
                attn_sim = torch.matmul(normed, normed.transpose(1, 2))
                attn_sim = torch.softmax(attn_sim, dim=-1)
                snr_sim = compute_attention_snr(attn_sim, labels, node_mask)
                if log_group and log_step:
                    wandb.log({f"ASNR_{log_group}/dotprod_softmax_after_diffusion": snr_sim,
                               "step": self.config.current_step})
        # --- End diffusion integration ---

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            input_nodes = torch.nn.functional.linear(input_nodes, self.graph_encoder.embed_tokens.weight)

        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        if self.config.enable_diffusion:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=input_nodes,
                hidden_states=inner_states), attention_matching_loss, dummy_node_loss
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states), dummy_node_loss

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes



class GraphormerForNodeClassification(GraphormerPreTrainedModel):
    """
    This model can be used for node-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per node
    - single-task classification (by setting config.num_classes to the number of classes); there should be one integer label per node
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list of integer labels for each node.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__(config)
        self.config = config
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier = GraphormerDecoderHead(self.embedding_dim, self.num_classes)
        self.is_encoder_decoder = True
        self.learnt_loss_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.learnt_attention_matching_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.learnt_dummy_node_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        attn_bias: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        edge_index: Optional[torch.LongTensor] = None,
        node_mask: Optional[torch.BoolTensor] = None,
        attn_override: Optional[dict] = None,
        **kwargs
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if len(node_mask.shape) == 1:
            node_mask = node_mask.unsqueeze(0)#.repeat(input_nodes.shape[0])  # add batch dimension
        encoder_outputs = self.encoder(
            input_nodes,
            input_edges,
            attn_bias,
            in_degree,
            out_degree,
            spatial_pos,
            attn_edge_type,
            return_dict=True,
            edge_index=edge_index,
            labels=labels,
            node_mask=node_mask,
            attn_override=attn_override,
            **kwargs
        )
        if self.config.enable_diffusion:
            encoder_outputs, attention_matching_loss, dummy_node_loss = encoder_outputs
        else:
            encoder_outputs, dummy_node_loss = encoder_outputs
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        # outputs: [batch, num_nodes+1, hidden_dim] (first token is graph token)
        node_outputs = outputs[:, 1:, :]  # remove graph token
        logits = self.classifier(node_outputs)  # [batch, num_nodes, num_classes]
        
        loss = None
        if labels is not None:
            # labels: [batch, num_nodes] or [batch, num_nodes, num_classes]
            if node_mask is not None:  # node mask required for train/val/test masks. graph has all
                # print("MASK:", node_mask.device, "LABELS:", torch.isnan(labels).device)
                mask = node_mask & ~torch.isnan(labels)
            else:
                mask = ~torch.isnan(labels)

            if self.num_classes == 1:  # regression
                loss_fct = L1Loss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 2:  # single-task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:  # binary multi-task classification
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])
            loss = loss * self.learnt_loss_scale
        
        loss += dummy_node_loss * self.learnt_dummy_node_scale

        if self.config.enable_diffusion:
            if np.random.rand() < 0.01:
                with open(f"{self.config.experiment_dir}/losses_node.csv", "a") as f:
                    print(f"{datetime.datetime.now()},{loss},{attention_matching_loss}", file=f)
            
            if isinstance(loss, torch.Tensor) and isinstance(attention_matching_loss, torch.Tensor):
                loss = loss + (attention_matching_loss * self.config.structure_scale * self.learnt_attention_matching_scale)
            else:
                loss = loss + attention_matching_loss
        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)

__all__ = [
    "GraphormerForGraphClassification",
    "GraphormerForNodeClassification",
    "GraphormerModel",
    "GraphormerPreTrainedModel"
]
