# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

import os
import torch
import torch.nn as nn
import numpy as np

from ..base import BaseModel, BaseConfig
from ..utils.base_utils import safe_getattr, safe_hasattr
from ..modules import init_graphormer_params
from ..modules import GraphormerGraphEncoder as GraphormerGraphEncoderBase
from ..modules import Graph3DBias, NodeTaskHead
from .graphormer_encoder import GraphormerEncoder as GraphormerEncoderBase

from .model_utils import (
    mask_after_k_persample,
    make_masks,
    get_center_pos,
    tensor_merge,
)

from .diffusion.schedulers.legacy_scheduler import get_beta_schedule

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class TimeStepEncoder(nn.Module):
    def __init__(self, args):
        super(TimeStepEncoder, self).__init__()
        self.args = args

        if args.diffusion_timestep_emb_type == "positional":
            self.time_proj = SinusoidalPositionEmbeddings(args.encoder_embed_dim)
        else:
            raise NotImplementedError

        self.time_embedding = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
            nn.GELU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )

    def forward(self, timesteps):
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        return t_emb


class GraphormerGraphEncoder(GraphormerGraphEncoderBase):
    """
    User-defined extra node layers or bias layers
    """

    def init_extra_node_layers(self, args):
        super().init_extra_node_layers(args)
        self.timestep_encoder = TimeStepEncoder(args)
        self.dist_feature_node_scale = args.dist_feature_node_scale
        self.tag_embedding = nn.Embedding(3, args.encoder_embed_dim)
        self.no_diffusion = args.no_diffusion

    def init_extra_bias_layers(self, args):
        super().init_extra_bias_layers(args)

    def forward_extra_node_layers(self, batched_data, x):
        x = super().forward_extra_node_layers(batched_data, x)
        assert x.shape[0] == batched_data["ts"].shape[0]

        if not self.no_diffusion:
            # get time embedding
            ts = batched_data["ts"].to(x.device)  # [B, ]
            time_emb = self.timestep_encoder(ts).type_as(x)  # [B, d]
            batched_data["time_emb"] = time_emb

            # add time embedding
            x += time_emb[:, None, :]
        else:
            batched_data["time_emb"] = None

        # add edge feature
        x[:, 1:, :] += batched_data["_edge_features"] * self.dist_feature_node_scale / 2
        x[:, 1:, :] += batched_data["_edge_features_crystal"] * self.dist_feature_node_scale / 2 # need to mask ligand part

        # add tag embedding
        atoms = batched_data["x"][:, :, 0]
        padding_mask = atoms.eq(0)
        n_graph, n_node = atoms.size()[:2]
        lnode = batched_data["lnode"]
        tag_mask = ~mask_after_k_persample(n_graph, n_node, lnode)
        tag_mask = tag_mask.masked_fill(padding_mask, 2).long()
        tag_features = self.tag_embedding(tag_mask)
        x[:, 1:, :] += tag_features

        batched_data["_edge_features"] = None
        return x

    def forward_extra_bias_layers(self, batched_data, attn_bias):
        bias = super().forward_extra_bias_layers(batched_data, attn_bias)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + batched_data["_attn_bias_3d"]

        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + batched_data["_attn_bias_3d_crystal"] # need to mask ligand part
        bias /= 3
        batched_data["_attn_bias_3d_crystal"] = None
        batched_data["_bias"] = bias
        batched_data["_attn_bias_3d"] = None
        return bias


class GraphormerEncoder(GraphormerEncoderBase):
    def build_graph_encoder(self, args):
        return GraphormerGraphEncoder(args)


@dataclass
class GraphormerDiffModelConfig(BaseConfig):
    num_diffusion_timesteps: int = field(
        default=5000, metadata={"help": "number of diffusion timesteps"}
    )

    diffusion_timestep_emb_type: str = field(
        default="positional",
        metadata={"help": "type of time embedding for diffusion timesteps"},
    )

    diffusion_layer_add_time_emb: bool = field(
        default=False,
        metadata={"help": "whether to add time embedding to node features layerwise"},
    )

    diffusion_layer_proj_time_emb: bool = field(
        default=False,
        metadata={"help": "whether to project time embedding layerwise"},
    )

    diffusion_beta_schedule: str = field(
        default="sigmoid", metadata={"help": "beta schedule for diffusion"}
    )
    diffusion_beta_start: float = field(
        default=1.0e-7, metadata={"help": "beta start for diffusion"}
    )
    diffusion_beta_end: float = field(
        default=2.0e-3, metadata={"help": "beta end for diffusion"}
    )
    diffusion_sampling: str = field(
        default="ddpm",
        metadata={"help": "sampling strategy, ddpm or ddim"},
    )
    ddim_steps: int = field(
        default=50,
        metadata={"help": "sampling steps for ddim"},
    )
    ddim_eta: float = field(
        default=0.0,
        metadata={"help": "eta for ddim"},
    )
    num_atom_types: int = field(
        default=128,
        metadata={"help": "number of atom types"},
    )
    dist_feature_extractor: str = field(
        default="rbf",
        metadata={"help": "distance feature extractor, can be rbf or gbf"},
    )
    dist_feature_node_scale: float = field(
        default=1.0,
        metadata={"help": "scale of distance feature added to node representations"},
    )
    dist_feature_num_kernels: int = field(
        default=128,
        metadata={"help": "number of kernels to extract distance features"},
    )
    prior_distribution_std: float = field(
        default=1,
        metadata={"help": "variance of prior distribution"},
    )
    reweighting_file: str = field(
        default="",
        metadata={
            "help": "using reweighting file to reweight the loss according to RMSD_to_crystal"
        },
    )
    no_diffusion: bool = field(
        default=False, metadata={"help": "disable diffusion and training on bare graphormer"},
    )
    ligand_only: bool = field(
        default=False, metadata={"help": "using water-ligand dataset"},
    )
    protein_only: bool = field(
        default=False, metadata={"help": "using protein only dataset"},
    )
    ligand_center: bool = field(
        default=False, metadata={"help": "using ligand center instead of protein center"},
    )
    pairwise_loss: bool = field(
        default=False, metadata={"help": "reweighting loss using atom numbers per protein-ligand pair"},
    )
    test_mode: bool = field(
        default=False, metadata={"help": "switch to test mode to generate conformers"},
    )
    num_epsilon_estimator: int = field(
        default=8, metadata={"help": "number of epsilons to sampled for trace estimation in flow ode"},
    )


class GraphormerDiffModel(BaseModel):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.init_diffusion(args)

    def max_nodes(self):
        return self.encoder.max_nodes()

    @classmethod
    def build_model(cls, args, task):
        encoder = GraphormerEncoder(args)
        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        encoder_out = self.encoder(batched_data, **kwargs)
        return encoder_out

    def init_diffusion(self, args):
        self.num_diffusion_timesteps = args.num_diffusion_timesteps
        self.diffusion_beta_schedule = args.diffusion_beta_schedule
        self.diffusion_beta_start = args.diffusion_beta_start
        self.diffusion_beta_end = args.diffusion_beta_end
        self.diffusion_sampling = args.diffusion_sampling
        self.ddim_steps = args.ddim_steps
        self.ddim_eta = args.ddim_eta
        self.no_diffusion = args.no_diffusion
        self.test_mode = args.test_mode
        self.num_epsilon_estimator = args.num_epsilon_estimator

        if not self.no_diffusion:
            self.betas = get_beta_schedule(
                self.diffusion_beta_schedule,
                self.num_diffusion_timesteps,
                self.diffusion_beta_start,
                self.diffusion_beta_end,
            )
            self.alphas = 1 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if not self.no_diffusion:
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
            self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
            self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
            self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)

    def get_sampling_output(
        self, batched_data, pos_center=None, sampling_times=1, **kwargs
    ):
        if self.no_diffusion:
            return self.forward(batched_data, **kwargs)
        else:
            if self.diffusion_sampling == "ddpm":
                return self.ddpm_sampling(batched_data, pos_center, sampling_times, **kwargs)
            elif self.diffusion_sampling == "ddim":
                return self.ddim_sampling(batched_data, pos_center, sampling_times, **kwargs)
            else:
                raise NotImplementedError

    def rigid_transform_Kabsch_3D_torch4batch(self, pred, orig, lig_mask, pro_mask):
        def rigid_transform_Kabsch_3D_torch(A, B):
            # R = 3x3 rotation matrix, t = 3x1 column vector
            # This already takes residue identity into account.
            assert A.shape[1] == B.shape[1]
            num_rows, num_cols = A.shape
            if num_rows != 3:
                raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

            num_rows, num_cols = B.shape
            if num_rows != 3:
                raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

            # find mean column wise
            centroid_A = torch.mean(A, dim=1, keepdim=True)
            centroid_B = torch.mean(B, dim=1, keepdim=True)

            # subtract mean
            Am = A - centroid_A
            Bm = B - centroid_B

            # dot is matrix multiplication for array
            H = Am @ Bm.T

            # find rotation
            U, S, Vt = torch.linalg.svd(H)

            R = Vt.T @ U.T

            # special reflection case
            if torch.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            t = -R @ centroid_A + centroid_B

            return R, t

        R_list = []
        t_list = []
        for i in range(pred.shape[0]):
            R, t = rigid_transform_Kabsch_3D_torch(
                pred[i, :, lig_mask[i]].T, orig[i, :, lig_mask[i]].T
            )
            R_list.append(R)
            t_list.append(t)
        R = torch.stack(R_list, dim=0)
        t = torch.stack(t_list, dim=0)
        return R, t
