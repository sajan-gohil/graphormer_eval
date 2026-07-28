from torch_geometric.graphgym.register import register_config

from graphgps.config.posenc_config import CN


@register_config('hop_masked')
def hop_masked_cfg(cfg):
    """Config group for the hop-masked transformer that replaces spatial GNN
    layers in the S^2GNN architecture.

    When ``cfg.gnn.hop_masked.enable`` is True the loader precomputes (or loads
    cached) Floyd-Warshall distance masks, and the network uses
    ``HopMaskedS2GNN`` which pairs a hop-masked transformer with the spectral
    layer at each message-passing step.
    """

    cfg.gnn.hop_masked = CN()

    # Master switch — set to True in the YAML to activate.
    cfg.gnn.hop_masked.enable = False

    # Transformer hidden dimension (node embedding width inside transformer).
    cfg.gnn.hop_masked.hidden_dim = 240

    # Number of stacked transformer sub-layers *within* each combined
    # [transformer + spectral] layer.  ``cfg.gnn.layers_mp`` controls how many
    # combined layers are stacked.
    cfg.gnn.hop_masked.num_layers = 1

    # K — total hop levels in the precomputed distance masks.
    cfg.gnn.hop_masked.num_hops = 30

    # Total attention heads.  In single mode this must equal
    # (num_hops - 1) + num_global_heads.
    cfg.gnn.hop_masked.num_heads = 30

    # FFN inner-dim multiplier (ffn_dim = hidden_dim * ffn_ratio).
    cfg.gnn.hop_masked.ffn_ratio = 1
