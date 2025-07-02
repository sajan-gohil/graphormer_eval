import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import erdos_renyi_graph


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
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.out = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.out(x)
        return x

class GraphLatentDiffusion(nn.Module):
    def __init__(self, input_dim=768, latent_dim=768, num_denoising_steps=50):
        super().__init__()
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
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy_x, noise

    def denoising_step(self, noisy_embeddings, t, edge_index):
        pred_noise = self.denoiser(noisy_embeddings, edge_index)
        return pred_noise

    def attention_improvement_loss(self, node_embeddings, denoised_embeddings, edge_index):
        src, dst = edge_index
        initial_scores = (node_embeddings[src] * node_embeddings[dst]).sum(dim=-1)
        final_scores = (denoised_embeddings[src] * denoised_embeddings[dst]).sum(dim=-1)
        # Encourage final_scores > initial_scores → hinge loss
        loss = F.relu(1.0 - (final_scores - initial_scores)).mean()
        return loss

    def forward(self, node_embeddings, edge_index):
        N = node_embeddings.shape[0]
        t = torch.randint(0, self.num_denoising_steps, (N,), device=node_embeddings.device)
        noisy_embeddings, true_noise = self.add_noise(node_embeddings, t)
        pred_noise = self.denoising_step(noisy_embeddings, t, edge_index)

        # MSE loss for noise prediction
        # mse_loss = F.mse_loss(pred_noise, true_noise)

        # Reconstruct denoised embeddings
        alpha_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        denoised_embeddings = (noisy_embeddings - one_minus_alpha_t * pred_noise) / alpha_t

        # Attention improvement loss
        attn_loss = self.attention_improvement_loss(node_embeddings, denoised_embeddings, edge_index)
        return attn_loss.item()


if __name__ == "__main__":
    # Parameters
    num_nodes = 100
    input_dim = 32
    latent_dim = 64
    num_steps = 20
    node_embeddings = torch.randn(num_nodes, input_dim)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.1)
    model = GraphLatentDiffusion(input_dim=input_dim, latent_dim=latent_dim, num_denoising_steps=num_steps)
    attn_loss = model(node_embeddings, edge_index)
    print(f"Attention improvement loss: {attn_loss:.4f}")