import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
import datetime


class DenoiserModel(nn.Module):
    def __init__(self, in_channels, time_embedding, num_layers=3, heads=4, layer_type="gat", config=None):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.time_embedding = time_embedding
        self.timestep_sise = time_embedding.shape[-1]
        self.num_layers = num_layers
        # Increase dim only for linear/GNN
        self.num_channels = [
            self.in_channels * i if layer_type != "mha" else self.in_channels
            for i in range(1, num_layers + 2)
        ]
        self.num_channels += self.num_channels[-2::-1]
        print(f"MAKING {layer_type} DENOISER WITH LAYER DIMS: {self.num_channels}")
        self.layer_type = layer_type

        self.norms = nn.ModuleList([
            nn.LayerNorm(self.num_channels[i])
            for i in range(1, len(self.num_channels))
        ])

        self.t_proj = nn.ModuleList([
            nn.Linear(self.timestep_sise, self.num_channels[i])
            for i in range(len(self.num_channels[:-1]))
        ])

        if layer_type == "linear":
            self.layers = nn.ModuleList([
                nn.Linear(self.num_channels[i], self.num_channels[i + 1])
                for i in range(len(self.num_channels[:-1]))
            ])
        elif layer_type == "gat":
            self.layers = nn.ModuleList([
                GATv2Conv(self.num_channels[i], self.num_channels[i + 1]//heads, heads=heads)
                for i in range(len(self.num_channels[:-1]))
            ])
        elif layer_type == "mha":
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=self.num_channels[i],
                                           nhead=heads,
                                           dim_feedforward=self.num_channels[i + 1],
                                           batch_first=True)
                for i in range(len(self.num_channels[:-1]))
            ])


    def forward(self, x_batch, edge_index_list):
        if self.layer_type == "gat":
            B, N, Feat = x_batch.shape
            data_list = [Data(x=x_batch[b], edge_index=edge_index_list[b]) for b in range(B)]
            batch = Batch.from_data_list(data_list)
            x_batch = batch.x
            edge_index_list = batch.edge_index
            time_embedding_batch = self.time_embedding[batch.batch]
        else:
            time_embedding_batch = self.time_embedding

        down_res = []
        for i in range(self.num_layers):
            print("=-=-=-=-", x_batch.shape, self.t_proj[i](time_embedding_batch).unsqueeze(1).shape)
            t_emb = self.t_proj[i](time_embedding_batch)
            if len(x_batch.shape) == 3:
                t_emb = t_emb.unsqueeze(1)
            x_batch += t_emb
            x_batch = self.layers[i](x_batch) if self.layer_type != "gat" else self.layers[i](
                x_batch, edge_index_list)
            if self.layer_type != "mha":
                x_batch = self.norms[i](x_batch)
            down_res.append(x_batch)

        down_res = down_res[::-1]
        for idx, i in enumerate(list(range(self.num_layers, len(self.layers))), 1): # Start from 1 to skip bottleneck
            t_emb = self.t_proj[i](time_embedding_batch)
            if len(x_batch.shape) == 3:
                t_emb = t_emb.unsqueeze(1)
            x_batch += t_emb
            x_batch = self.layers[i](x_batch) if self.layer_type != "gat" else self.layers[i](
                x_batch, edge_index_list)
            if idx < len(down_res):
                x_batch += down_res[idx]
            # elif down_res[idx].shape == x_batch.shape:
            #     x_batch += down_res[idx]
            if self.layer_type != "mha":
                x_batch = self.norms[i](x_batch)
        return x_batch

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data, Batch

    def make_synthetic_batch(batch_size=2, num_nodes=5, in_channels=8):
        data_list = []
        for b in range(batch_size):
            x = torch.randn(num_nodes, in_channels)  # node features
            # simple ring graph
            edge_index = torch.tensor([
                [i for i in range(num_nodes)] + [(i + 1) % num_nodes for i in range(num_nodes)],
                [(i + 1) % num_nodes for i in range(num_nodes)] + [i for i in range(num_nodes)],
            ], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index))
        batch = Batch.from_data_list(data_list)
        return batch

    def main():
        in_channels = 8
        timestep_dim = 16
        num_layers = 2
        # fake embedding layer for timesteps (like nn.Embedding)
        # simulate B=2 with random timestep ids
        batch_size = 2
        timesteps = torch.randint(0, 1000, (batch_size,))
        timestep_embedding = torch.nn.Embedding(1000, timestep_dim)(timesteps)  # (B, timestep_dim)

        # make batch of graphs
        batch = make_synthetic_batch(batch_size=batch_size, num_nodes=5, in_channels=in_channels)

        # prepare input: reshape to (B,N,F) for non-GAT, or keep Batch for GAT
        x_batch = torch.stack([d.x for d in batch.to_data_list()])  # (B,N,F)
        edge_index_list = [d.edge_index for d in batch.to_data_list()]

        # create model (test with GAT)
        model = DenoiserModel(
            in_channels=in_channels,
            time_embedding=timestep_embedding,
            num_layers=num_layers,
            heads=4,
            layer_type="mha",
        )
        print("Input shape:", x_batch.shape)
        # forward pass
        out = model(x_batch, edge_index_list)
        print("Output shape:", out.shape)
    main()