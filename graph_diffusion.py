import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
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
            edge_index_list: list of [2, E_i] tensors
        Returns:
            Tensor of shape [B, N, out_features]
        """
        B, N, F = x_batch.shape
        data_list = []

        for b in range(B):
            data = Data(x=x_batch[b], edge_index=edge_index_list[b])
            data_list.append(data)

        batch = Batch.from_data_list(data_list)  # Automatically handles indexing

        x = torch.nn.functional.elu(self.gat1(batch.x, batch.edge_index))
        x = torch.nn.functional.elu(self.gat2(x, batch.edge_index))
        x = self.out(x)
        out_per_graph = x.split(batch.batch.bincount().tolist(), dim=0)
        return torch.stack(out_per_graph, dim=0)  # Shape: [B, N, out_features] if N is fixed


class GraphLatentDiffusion(nn.Module):
    def __init__(self, input_dim=768, latent_dim=768, num_denoising_steps=50):
        super().__init__()
        print("MAKING LATENT DIFFUSION MODEL")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_denoising_steps = num_denoising_steps
        self.timestep_embeddings = nn.Embedding(num_denoising_steps, latent_dim)
        
        betas = cosine_beta_schedule(num_denoising_steps)
        self.register_buffer("betas", betas)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.denoiser = GATv2Denoiser(input_dim, latent_dim, input_dim)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1)#.unsqueeze(2)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)#.unsqueeze(2)
        # print("Noise addition shapes", sqrt_alpha.shape, x.shape, sqrt_one_minus_alpha.shape, noise.shape)
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        # print("NOISY X SHAPE = ", noisy_x.shape)
        return noisy_x, noise

    # def denoising_step(self, noisy_embeddings, t, edge_index):
    #     pred_noise = self.denoiser(noisy_embeddings, edge_index)
    #     return pred_noise

    def attention_improvement_loss(self, node_embeddings, denoised_embeddings, edge_index):
        #print(torch.tensor(edge_index).shape)
        #print(len(edge_index), [i.shape for i in edge_index])
        losses = []
        for src, dst in edge_index:
            # src, dst = edge_index
            initial_scores = torch.nn.sigmoid(node_embeddings[src] * node_embeddings[dst]).sum(dim=-1)
            final_scores = torch.nn.sigmoid(denoised_embeddings[src] * denoised_embeddings[dst]).sum(dim=-1)
        
            # Encourage final_scores > initial_scores -> hinge loss
            losses.append(torch.nn.functional.relu(1.0 - (final_scores - initial_scores)).mean())
        # print(losses)
        return torch.tensor(losses).mean()

    def forward(self, node_embeddings, edge_index):
        N = node_embeddings.shape[0]
        t = torch.randint(0, self.num_denoising_steps, (N,), device=node_embeddings.device)
        noisy_embeddings, true_noise = self.add_noise(node_embeddings, t)
        # pred_noise = self.denoising_step(noisy_embeddings, t, edge_index)
        # denoised_embeddings = self.denoising_step(noisy_embeddings, t, edge_index)
        denoised_embeddings = self.denoiser(noisy_embeddings, edge_index)

        # MSE loss for noise prediction
        # mse_loss = F.mse_loss(pred_noise, true_noise)

        # Reconstruct denoised embeddings
        # alpha_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        # one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        # denoised_embeddings = (noisy_embeddings - one_minus_alpha_t * pred_noise) / alpha_t

        # Attention improvement loss
        attn_loss = 0 # self.attention_improvement_loss(node_embeddings, denoised_embeddings, edge_index)
        return denoised_embeddings, attn_loss #.item()

    @torch.no_grad()
    def sample(self, noisy_embeddings, edge_index):
        x = noisy_embeddings
        for t in reversed(range(self.num_denoising_steps)):
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            pred_noise = self.denoising_step(x, t_tensor, edge_index)

            alpha_t = self.sqrt_alphas_cumprod[t].view(1, 1)
            one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1)

            x = (x - one_minus_alpha_t * pred_noise) / alpha_t
            # optionally add noise at intermediate steps, if stochastic sampling desired
        return x



if __name__ == "__main__":
    from torch_geometric.utils import erdos_renyi_graph
    # Parameters
    num_nodes = 100
    input_dim = 32
    latent_dim = 64
    num_steps = 20
    node_embeddings = torch.randn(num_nodes, input_dim)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.1)
    model = GraphLatentDiffusion(input_dim=input_dim, latent_dim=latent_dim, num_denoising_steps=num_steps)
    denoised_embeddings, attn_loss = model(node_embeddings, edge_index)
    print(f"Attention improvement loss: {attn_loss.item():.4f}")
