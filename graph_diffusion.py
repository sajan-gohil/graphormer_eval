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


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, steps=timesteps)


class GATv2Denoiser(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=4, use_linear=False, **kwargs):
        super().__init__()
        print("INITIALIZING GENERALIZED DENOISER")
        self.use_linear = use_linear
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.down_blocks = nn.ModuleList()
        self.down_norms = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_norms = nn.ModuleList()

        # Build downsampling path
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            out_dim = hidden_channels
            if use_linear:
                layer = nn.Linear(in_dim, out_dim)
            else:
                layer = GATv2Conv(in_dim, out_dim, heads=heads)
            self.down_blocks.append(layer)
            self.down_norms.append(nn.LayerNorm(out_dim * heads if not use_linear else out_dim))

        # Build upsampling path (same number of layers, reverse direction)
        for i in range(num_layers):
            in_dim = hidden_channels * heads
            out_dim = hidden_channels
            if use_linear:
                layer = nn.Linear(in_dim, out_dim)
            else:
                layer = GATv2Conv(in_dim, out_dim, heads=heads)
            self.up_blocks.append(layer)
            self.up_norms.append(nn.LayerNorm(out_dim * heads if not use_linear else out_dim))

        final_in = hidden_channels * heads if not use_linear else hidden_channels
        self.output_layer = nn.Linear(final_in, out_channels)
        self.output_norm = nn.LayerNorm(out_channels)

    def forward(self, x_batch, edge_index_list):
        B, N, F = x_batch.shape
        data_list = [Data(x=x_batch[b], edge_index=edge_index_list[b]) for b in range(B)]
        batch = Batch.from_data_list(data_list)
        x = batch.x
        edge_index = batch.edge_index
        skip_connections = []
        # Down path
        for i in range(self.num_layers):
            layer = self.down_blocks[i]
            norm = self.down_norms[i]
            if self.use_linear:
                x = F.elu(norm(layer(x)))
            else:
                x = F.elu(norm(layer(x, edge_index)))
            skip_connections.append(x)
        # Up path
        for i in range(self.num_layers):
            layer = self.up_blocks[i]
            norm = self.up_norms[i]
            skip_x = skip_connections[-(i + 2)] if i < self.num_layers - 1 else torch.zeros_like(x)
            x = x + skip_x  # residual from down path
            if self.use_linear:
                x = F.elu(norm(layer(x)))
            else:
                x = F.elu(norm(layer(x, edge_index)))
        # Final projection
        x = self.output_norm(self.output_layer(x))
        # Split batched graph output
        out_per_graph = x.split(batch.batch.bincount().tolist(), dim=0)
        return torch.stack(out_per_graph, dim=0)  # [B, N, out_channels]


class GraphLatentDiffusion(nn.Module):
    def __init__(self, input_dim=768, latent_dim=768, num_denoising_steps=100, config=None):
        super().__init__()
        print("MAKING LATENT DIFFUSION MODEL")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_denoising_steps = num_denoising_steps
        self.config = config
        self.reconstruction_scale = getattr(config, "reconstruction_scale", 0.5)
        self.structure_scale = getattr(config, "structure_scale", 0.1)
        self.timestep_embeddings = nn.Embedding(num_denoising_steps, latent_dim)
        
        if config.diffusion_type == "ddim":
            betas = linear_beta_schedule(num_denoising_steps)
        else:
            betas = cosine_beta_schedule(num_denoising_steps)
        # betas = linear_beta_schedule(num_denoising_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.denoiser = GATv2Denoiser(input_dim+latent_dim, latent_dim//4, input_dim, heads=4)
        self.diffusion_optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=1e-4)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy_x, noise

    def predict_x0_from_noise(self, noisy_x, noise_pred, t):
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        return (noisy_x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

    def optimize_diffusion(self, node_embeddings, edge_index_list):
        """
        Optimize only the diffusion model over all timesteps for this batch.
        """
        B = node_embeddings.size(0)
        t = torch.tensor([self.num_denoising_steps - 1], device=node_embeddings.device).repeat(B)
        noisy_x, true_noise = self.add_noise(node_embeddings.detach().clone(), t)

        x_t = noisy_x.clone()

        for step in reversed(range(self.num_denoising_steps)):
            x_t = x_t.detach()
            t_step = torch.tensor([step], device=node_embeddings.device).repeat(B)
            t_emb = self.timestep_embeddings(t_step).unsqueeze(1).expand(-1, x_t.size(1), -1)

            noisy_with_t = torch.cat([x_t, t_emb], dim=-1)
            noise_pred = self.denoiser(noisy_with_t, edge_index_list)

            loss = MSELoss()(true_noise, noise_pred)
            if self.config.optimize_diffuser:
                # print("TRYING ========")
                self.diffusion_optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.diffusion_optimizer.step()
                # print("====optimized")

            # Update x_t -> x_{t-1} (DDIM-like deterministic step)
            x0_pred = self.predict_x0_from_noise(x_t, noise_pred, t_step)  # .detach().clone()
            if step > 0:
                alpha_prev = self.alphas_cumprod[step - 1]
                x_t = torch.sqrt(alpha_prev).unsqueeze(0).unsqueeze(-1) * x0_pred + \
                      torch.sqrt(1 - alpha_prev).unsqueeze(0).unsqueeze(-1) * noise_pred

        return x_t  # final denoised embeddings after training

    
    def sample_diffusion(self, node_embeddings, edge_index_list):
        """
        Run deterministic DDIM-like sampling (no optimizer update).
        """
        B = node_embeddings.size(0)
        t = torch.tensor([self.num_denoising_steps - 1], device=node_embeddings.device).repeat(B)
        noisy_x, _ = self.add_noise(node_embeddings.detach(), t)

        x_t = noisy_x.clone()

        for step in reversed(range(self.num_denoising_steps)):
            t_step = torch.tensor([step], device=node_embeddings.device).repeat(B)
            t_emb = self.timestep_embeddings(t_step).unsqueeze(1).expand(-1, x_t.size(1), -1)

            noisy_with_t = torch.cat([x_t, t_emb], dim=-1)
            noise_pred = self.denoiser(noisy_with_t, edge_index_list)

            x0_pred = self.predict_x0_from_noise(x_t, noise_pred, t_step)
            if step > 0:
                alpha_prev = self.alphas_cumprod[step - 1]
                x_t = torch.sqrt(alpha_prev).unsqueeze(0).unsqueeze(-1) * x0_pred + \
                      torch.sqrt(1 - alpha_prev).unsqueeze(0).unsqueeze(-1) * noise_pred
            else:
                x_t = x0_pred

        return x_t

    def attention_improvement_loss(self,
                                   node_embeddings,
                                   denoised_embeddings,
                                   edge_index_list,
                                   # threshold=0.1,
                                   tau=0.2):
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
        if np.random.rand() < 0.1:
            with open(f"{self.config.experiment_dir}/structural_associations.csv",
                      "a") as f:
                struct_assn = self.calculate_structural_associations(flat_node.detach().cpu(),
                                                         flat_denoised.detach().cpu(),
                                                         all_src, all_dst)
                print(f"{datetime.datetime.now()},{struct_assn[0]},{struct_assn[1]}", file=f)
        initial_scores = (flat_node[all_src] * flat_node[all_dst]).sum(-1)
        final_scores = (flat_denoised[all_src] * flat_denoised[all_dst]).sum(-1)
        
        with torch.no_grad():
            threshold = torch.zeros_like(initial_scores)
            for b in range(node_embeddings.size(0)):
                mask = (all_batch == b)
                threshold[mask] = initial_scores[mask].mean()

        recall_init = torch.sigmoid((initial_scores - threshold) / tau)
        recall_final = torch.sigmoid((final_scores - threshold) / tau)

        per_graph_loss = torch.zeros(node_embeddings.size(0), device=node_embeddings.device)
        # per_graph_loss.index_add_(0, all_batch, -(recall_final - recall_init))
        per_graph_loss.index_add_(0, all_batch, -torch.log(recall_final+1e-8))
        # per_graph_loss.index_add_(0, all_batch, F.relu(1 - (final_scores - initial_scores)))
        num_graphs = node_embeddings.size(0)
        counts = torch.bincount(all_batch, minlength=num_graphs).float()
        counts[counts == 0] = 1  # avoid division by zero
        return (per_graph_loss / counts).mean()

    def calculate_structural_associations(self, flat_node, flat_denoised, all_src,
                                          all_dst):
        with torch.no_grad():
            mask = torch.zeros((flat_node.shape[0], flat_node.shape[0]), dtype=torch.int8)
            mask[all_src, all_dst] = 1
            # mask_sum = mask.sum()
            mask_shape = mask.shape[0]**2
            node_adj = torch.mm(flat_node, flat_node.T)
            denoised_adj = torch.mm(flat_denoised, flat_denoised.T)
            max_node_recovery = 0
            max_denoised_recovery = 0
            for thresh in np.linspace(node_adj.mean() - node_adj.std(),
                                      node_adj.mean() + node_adj.std(), 9):
                adj = (node_adj > thresh).to(dtype=torch.int8)
                # print("METRI adj=", adj.shape, "Mask=", mask.shape)
                max_node_recovery = max((adj == mask).sum()/mask_shape,
                                        max_node_recovery)
            for thresh in np.linspace(denoised_adj.mean() - denoised_adj.std(),
                                      denoised_adj.mean() + denoised_adj.std(), 9):
                adj = (denoised_adj > thresh).to(dtype=torch.int8)
                max_denoised_recovery = max((adj == mask).sum()/mask_shape,
                                        max_denoised_recovery)
        return max_node_recovery, max_denoised_recovery

    def log_embedding_distribution(self, node_embeddings, denoised_embeddings):
        if np.random.rand() < 0.01:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.hist(denoised_embeddings.detach().cpu().reshape(-1))
            plt.savefig(
                os.path.join(
                    self.config.experiment_dir, "denoised_emb_dist_" +
                    timestamp + ".png"))
            plt.clf()
            plt.hist(node_embeddings.detach().cpu().reshape(-1))
            plt.savefig(
                os.path.join(
                    self.config.experiment_dir, "node_emb_dist_" +
                    timestamp + ".png"))
            plt.clf()

    def forward(self, node_embeddings, edge_index_list, aug_added_edges=None, aug_removed_edges=None, aug_original_edges=None):
        # print("NODE EMBEDDINGS SHAPE = ", node_embeddings.shape)  # B, N, D
        B = node_embeddings.shape[0]
        t = torch.randint(0, self.num_denoising_steps-1, (B,), device=node_embeddings.device)
        t_emb = self.timestep_embeddings(t).unsqueeze(1).expand(-1, node_embeddings.size(1), -1)

        if self.config.detached_denoiser:
            noisy_embeddings, true_noise = self.add_noise(node_embeddings.detach().clone(), t)
        else:
            noisy_embeddings, true_noise = self.add_noise(node_embeddings, t)
        noisy_embeddings_with_t = torch.cat([noisy_embeddings, t_emb], dim=-1)
        
        if not self.config.diffusion_type == "ddim":
            denoised_embeddings = self.denoiser(noisy_embeddings_with_t, edge_index_list)
        
        if self.config.diffusion_type == "x0":
            reconstruction_loss = MSELoss()(node_embeddings, denoised_embeddings)
        elif self.config.diffusion_type == "delta":
            reconstruction_loss = MSELoss()(true_noise, denoised_embeddings)
            denoised_embeddings = node_embeddings + denoised_embeddings
        elif self.config.diffusion_type == "noise_pred":
            reconstruction_loss = MSELoss()(true_noise, denoised_embeddings)
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)  # Remove more noise than added
            denoised_embeddings = (noisy_embeddings - sqrt_one_minus_alpha*denoised_embeddings)/sqrt_alpha
           
            t_2 = torch.ones((B,), dtype=torch.long, device=node_embeddings.device)
            t_emb_2 = self.timestep_embeddings(t_2).unsqueeze(1).expand(-1, node_embeddings.size(1), -1)
            last_noise_pred = self.denoiser(torch.cat([denoised_embeddings, t_emb_2], dim=-1), edge_index_list)
           
            sqrt_alpha_1 = self.sqrt_alphas_cumprod[t_2].unsqueeze(1).unsqueeze(2)
            sqrt_one_minus_alpha_1 = self.sqrt_one_minus_alphas_cumprod[t_2].unsqueeze(1).unsqueeze(2)
            denoised_embeddings = (denoised_embeddings - sqrt_one_minus_alpha_1*last_noise_pred)/sqrt_alpha
            denoised_embeddings = (denoised_embeddings - denoised_embeddings.mean())/denoised_embeddings.std()
            # denoised_embeddings = (0.1*node_embeddings) + (0.9*denoised_embeddings)
        
        elif self.config.diffusion_type == "ddim":
            _ = self.optimize_diffusion(node_embeddings, edge_index_list)
            denoised_embeddings = self.sample_diffusion(node_embeddings, edge_index_list)
            denoised_embeddings = 0.5 * denoised_embeddings + 0.5 * node_embeddings
            reconstruction_loss = 0
            
        self.log_embedding_distribution(node_embeddings, denoised_embeddings)

        # Attention improvement loss
        attn_loss = 0
        if self.structure_scale > 0:
            attn_loss = self.attention_improvement_loss(node_embeddings, denoised_embeddings, edge_index_list)

        # Auxiliary edge attention loss (if augmentation info provided)
        aux_loss = 0
        if aug_added_edges is not None and aug_removed_edges is not None and aug_original_edges is not None:
            aux_loss = self.aux_edge_attention_loss(denoised_embeddings, aug_added_edges, aug_removed_edges, aug_original_edges)
        total_loss = (attn_loss*self.structure_scale) + (reconstruction_loss*self.reconstruction_scale)
        if aux_loss != 0:
            total_loss = total_loss + 0.1 * aux_loss  # weight for aux loss
        return denoised_embeddings, total_loss
        # return denoised_embeddings, (attn_loss*self.structure_scale) + (reconstruction_loss*self.reconstruction_scale)

    def aux_edge_attention_loss(self, denoised_embeddings, aug_added_edges, aug_removed_edges, aug_original_edges):
        """
        For each graph in batch:
        - For randomly added edges: attention score should be close to zero.
        - For randomly removed edges: attention score should be higher than for non-existing edges.
        """
        # denoised_embeddings: [B, N, D]
        # aug_added_edges, aug_removed_edges, aug_original_edges: list of tensors per graph
        losses = []
        margin = 0.0
        for b in range(denoised_embeddings.size(0)):
            emb = denoised_embeddings[b]  # [N, D]
            # Compute attention scores (dot product)
            attn_scores = torch.matmul(emb, emb.T)  # [N, N]
            # Clamp to [0,1] for interpretability
            attn_scores = torch.sigmoid(attn_scores)
            # Added edges: want attn ~ 0
            added = aug_added_edges[b]
            if added.numel() > 0:
                added_scores = attn_scores[added[:,0], added[:,1]]
                loss_added = (added_scores ** 2).mean()  # penalize nonzero
            else:
                loss_added = 0.0
            # Removed edges: want attn > non-edge mean + margin
            removed = aug_removed_edges[b]
            orig = aug_original_edges[b]
            if removed.numel() > 0:
                removed_scores = attn_scores[removed[:,0], removed[:,1]]
                # Compute mean of all non-edges (excluding original edges)
                N = emb.size(0)
                all_idx = torch.ones((N,N), dtype=torch.bool, device=emb.device)
                all_idx[orig[0], orig[1]] = False
                non_edge_scores = attn_scores[all_idx]
                if non_edge_scores.numel() > 0:
                    non_edge_mean = non_edge_scores.mean()
                else:
                    non_edge_mean = 0.0
                # Encourage removed edge attn to be higher than non-edge mean + margin
                loss_removed = torch.relu(non_edge_mean + margin - removed_scores).mean()
            else:
                loss_removed = 0.0
            losses.append(loss_added + loss_removed)
        return sum(losses) / max(1, len(losses))

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
