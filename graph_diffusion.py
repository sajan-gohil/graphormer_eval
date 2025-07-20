import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GATv2Denoiser(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        print("INITIALIZING DIFFUSION")
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.out = nn.Linear(hidden_channels * heads, out_channels)

    def sample_forward(self, x, edge_index):
        x = torch.nn.functional.elu(self.gat1(x, edge_index))
        x = torch.nn.functional.elu(self.gat2(x, edge_index))
        x = self.out(x)
        return x

    def forward(self, x_batch, edge_index_list):
        """
        Args:
            x_batch: Tensor of shape [B, N, F]
            edge_index_list: list of [3, E_i] tensors
        Returns:
            Tensor of shape [B, N, out_features]
        """
        B, N, F = x_batch.shape
        data_list = []

        for b in range(B):
            data = Data(x=x_batch[b], edge_index=edge_index_list[b])
            data_list.append(data)

        batch = Batch.from_data_list(data_list)  # Automatically handles indexing

        x1 = torch.nn.functional.elu(self.gat1(batch.x, batch.edge_index))
        x2 = torch.nn.functional.elu(self.gat2(x1, batch.edge_index))
        x = self.out(x2)
        out_per_graph = x.split(batch.batch.bincount().tolist(), dim=0)
        return torch.stack(out_per_graph, dim=0)  # Shape: [B, N, out_features] if N is fixed


class GraphLatentDiffusion(nn.Module):
    def __init__(self, input_dim=768, latent_dim=768, num_denoising_steps=50, config=None):
        super().__init__()
        print("MAKING LATENT DIFFUSION MODEL")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_denoising_steps = num_denoising_steps
        self.config = config
        self.reconstruction_scale = getattr(config, "reconstruction_scale", 0.0)
        self.timestep_embeddings = nn.Embedding(num_denoising_steps, latent_dim)
        
        betas = cosine_beta_schedule(num_denoising_steps)
        self.register_buffer("betas", betas)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.denoiser = GATv2Denoiser(input_dim+latent_dim, latent_dim, input_dim)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        # print("Noise addition shapes", sqrt_alpha.shape, x.shape, sqrt_one_minus_alpha.shape, noise.shape, t.shape, t)
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        # print("NOISY X SHAPE = ", noisy_x.shape)
        return noisy_x, noise

    def attention_improvement_loss_slow(self, node_embeddings, denoised_embeddings, edge_index_list):
        losses = []
        node_emb_normed = node_embeddings  # torch.nn.functional.normalize(node_embeddings, p=2, dim=-1)
        denoised_emb_normed = denoised_embeddings  # torch.nn.functional.normalize(denoised_embeddings, p=2, dim=-1)
        for batch_index, (src, dst) in enumerate(edge_index_list):
            initial_scores = (node_emb_normed[batch_index][src] *
                              node_emb_normed[batch_index][dst]).sum(dim=-1)
            final_scores = (denoised_emb_normed[batch_index][src] *
                            denoised_emb_normed[batch_index][dst]).sum(dim=-1)
            # Encourage final_scores > initial_scores -> hinge loss
            losses.append(torch.nn.functional.relu(1.0 - (final_scores - initial_scores)).mean())
        # print(losses)
        return torch.stack(losses).mean()
    
    def attention_improvement_loss(self,
                                   node_embeddings,
                                   denoised_embeddings,
                                   edge_index_list,
                                   threshold=0.0):
        # Normalize embeddings
        node_emb_normed = F.normalize(node_embeddings, p=2, dim=-1)
        denoised_emb_normed = F.normalize(denoised_embeddings, p=2, dim=-1)

        # Concatenate all graphs into a single tensor for faster processing
        all_src = []
        all_dst = []
        all_batch = []
        offset = 0
        for b, edge_index in enumerate(edge_index_list):
            src, dst = edge_index
            all_src.append(src + offset)
            all_dst.append(dst + offset)
            all_batch.append(torch.full((src.size(0),), b, device=src.device))
            offset += node_embeddings[b].size(0)
    
        all_src = torch.cat(all_src)
        all_dst = torch.cat(all_dst)
        all_batch = torch.cat(all_batch)
    
        # Flatten embeddings [B, N, D] -> [sum(N), D]
        flat_node = node_emb_normed.reshape(-1, node_emb_normed.size(-1))
        flat_denoised = denoised_emb_normed.reshape(-1, denoised_emb_normed.size(-1))
    
        initial_scores = (flat_node[all_src] * flat_node[all_dst]).sum(-1)
        final_scores = (flat_denoised[all_src] * flat_denoised[all_dst]).sum(-1)
        
        recall_init = torch.sigmoid((initial_scores - threshold)/0.05)
        recall_final = torch.sigmoid((final_scores - threshold)/0.05)

        per_graph_loss = torch.zeros(node_embeddings.size(0), device=node_embeddings.device)
        # per_graph_loss.index_add_(0, all_batch, -(recall_final - recall_init))
        per_graph_loss.index_add_(0, all_batch, -(recall_final))
        # per_graph_loss.index_add_(0, all_batch, F.relu(1 - (final_scores - initial_scores)))
    
        return (per_graph_loss / torch.bincount(all_batch).float()).mean()

    def calculate_structural_associations(flat_node, flat_denoised, all_src,
                                          all_dst):
        with torch.no_grad():
            mask = torch.zeros((flat_node, flat_node))
            mask[all_src, all_dst] = 1
            mask_sum = mask.sum()
            node_adj = torch.mm(flat_node, flat_node.T)
            denoised_adj = torch.mm(flat_denoised, flat_denoised.T)
            max_node_recovery = 0
            max_denoised_recovery = 0
            for thresh in np.linspace(node_adj.mean() - node_adj.std(),
                                      node_adj.mean() + node_adj.std(), 9):
                adj = node_adj > thresh
                max_node_recovery = max(torch.bitwise_and(adj, mask)/mask_sum,
                                        max_node_recovery)
            for thresh in np.linspace(denoised_adj.mean() - denoised_adj.std(),
                                      denoised_adj.mean() + denoised_adj.std(), 9):
                adj = denoised_adj > thresh
                max_denoised_recovery = max(torch.bitwise_and(adj, mask)/mask_sum,
                                        max_node_recovery)
        return max_node_recovery, max_denoised_recovery


    def forward(self, node_embeddings, edge_index_list):
        # print("NODE EMBEDDINGS SHAPE = ", node_embeddings.shape)  # B, N, D
        B = node_embeddings.shape[0]
        t = torch.randint(0, self.num_denoising_steps, (B,), device=node_embeddings.device)
        t_emb = self.timestep_embeddings(t).unsqueeze(1).expand(-1, node_embeddings.size(1), -1)

        noisy_embeddings, true_noise = self.add_noise(node_embeddings, t)
        noisy_embeddings_with_t = torch.cat([noisy_embeddings, t_emb], dim=-1)
        denoised_embeddings = self.denoiser(noisy_embeddings_with_t, edge_index_list)
        if np.random.rand() < 0.001:
            plt.plot(denoised_embeddings.detach().cpu().reshape(-1))
            plt.savefig(
                os.path.join(
                    config.experiment_dir, "denoised_emb_dist_" +
                    "".join(np.random.choice(["a", "b", "c"], size=10))+".png"))
        # If noise predictor:
        # sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        # sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        # denoised_embeddings = (noisy_embeddings - sqrt_one_minus_alpha*pred_noise)/sqrt_alpha

        # Attention improvement loss
        attn_loss = self.attention_improvement_loss(node_embeddings, denoised_embeddings, edge_index_list)
        reconstruction_loss = MSELoss()(node_embeddings, denoised_embeddings)
        return denoised_embeddings, attn_loss + (reconstruction_loss*self.reconstruction_scale)


if __name__ == "__main__":
    from torch_geometric.utils import erdos_renyi_graph
    # Parameters
    num_nodes = 100
    input_dim = 32
    latent_dim = 64
    num_steps = 100
    batch_size = 10
    node_embeddings = torch.randn(batch_size, num_nodes, input_dim)
    edge_index = [erdos_renyi_graph(num_nodes, edge_prob=0.1) for i in range(batch_size)]
    model = GraphLatentDiffusion(input_dim=input_dim, latent_dim=latent_dim, num_denoising_steps=num_steps)
    denoised_embeddings, attn_loss = model(node_embeddings, edge_index)
    print(f"Attention improvement loss: {attn_loss.item():.4f}")
