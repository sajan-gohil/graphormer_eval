"""
Configuration for Phase 1: Base Transformer Training.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    root: str = "./data"
    dataset_name: str = "Peptides-func"
    batch_size: int = 64
    num_workers: int = 4
    # Positional encodings
    lap_eigvec_k: int = 8          # Number of Laplacian eigenvectors
    rwse_walk_lengths: int = 20     # Random walk structural encoding max walk length


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 5
    num_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1
    local_gnn_type: str = "GIN"     # "GIN" or "PNA"
    num_classes: int = 10
    # Positional encoding dims
    lap_dim: int = 8                # Must match lap_eigvec_k
    rwse_dim: int = 20              # Must match rwse_walk_lengths
    pe_hidden_dim: int = 64         # Hidden dim for PE projection


@dataclass
class TrainingConfig:
    max_epochs: int = 300
    warmup_epochs: int = 10
    patience: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "auto"            # "auto", "cuda", "cpu"
    log_per_class_every: int = 10   # Log per-class AP every N epochs


@dataclass
class Phase1Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
