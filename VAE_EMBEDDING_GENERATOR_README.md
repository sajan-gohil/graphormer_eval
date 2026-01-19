# VAE-based Node Embedding Generator

## Overview
This implementation adds a Variational Autoencoder (VAE) based node embedding generator to the GraphTransformer model in `new_experiments/graph_transformer_lrgb.py`.

## Mathematical Framework
The implementation follows the framework: **f(G) = {EP(Ni | N-Ni) for every i = 0 to i = len(N)}**

Where:
- **G** = Graph
- **N** = All node embeddings
- **Ni** = i'th node embedding
- **EP(Ni | N-Ni)** = Embedding Predictor that predicts i'th node embedding using all other nodes
- **f(G)** = Process that refines graph embeddings

The goal is to improve task performance: **L(t(G)) > L(t(f(G)))**

## Components

### 1. NodeEmbeddingVAE
A Variational Autoencoder for node embeddings with:
- **Encoder**: Maps embeddings to latent distribution (mu, log_variance)
- **Reparameterization**: z = mu + eps * sigma
- **Decoder**: Generates refined embeddings from latent samples
- **No reconstruction loss**: Trained end-to-end with task loss only

### 2. NodeEmbeddingGenerator
Implements EP(Ni | N-Ni) for all nodes:
- For each node i in the graph:
  - Masks out the i'th node embedding
  - Aggregates information from all other nodes using attention
  - Uses VAE to predict/generate refined embedding
- Special handling for single-node graphs

### 3. GraphTransformer Integration
The VAE refinement is applied:
- **After** transformer layers process initial embeddings
- **Before** graph-level pooling
- **During** the forward pass

## Usage

### Basic Usage (VAE enabled by default)
```bash
python new_experiments/graph_transformer_lrgb.py --epochs 200 --hidden_dim 128
```

### Disable VAE Refinement
```bash
python new_experiments/graph_transformer_lrgb.py --no_vae_refiner --epochs 200
```

### All Available Arguments
```bash
python new_experiments/graph_transformer_lrgb.py \
  --epochs 200 \
  --num_layers 3 \
  --hidden_dim 128 \
  --num_heads 4 \
  --dropout 0.5 \
  --lr 3e-4 \
  [--no_vae_refiner]  # Optional: disable VAE refinement
```

## Code Example

```python
from new_experiments.graph_transformer_lrgb import (
    NodeEmbeddingVAE,
    NodeEmbeddingGenerator,
    GraphTransformer
)

# Create model with VAE refinement
model = GraphTransformer(
    in_dim=9,
    hidden_dim=128,
    out_dim=10,
    layers=3,
    heads=4,
    dropout=0.5,
    use_vae_refiner=True  # Enable VAE refinement (default)
)

# Forward pass
output = model(x, edge_index, batch)
# VAE refinement is applied automatically before pooling
```

## Training Considerations

### Loss Function
- **Only** binary cross-entropy loss (task loss)
- **No** reconstruction loss for the VAE
- Gradient flows through VAE to improve task performance

### Architecture
- Latent dimension is automatically set to `hidden_dim // 2`
- Context aggregation uses 4 attention heads (configurable)
- Dropout of 0.1 for context attention (configurable)

### Performance
- O(n²) complexity per graph where n = number of nodes
- For large graphs, this may be a bottleneck
- Future optimization could involve vectorized operations

## Testing

Run the comprehensive test suite:
```bash
python test_vae_embedding_generator.py
```

Tests include:
1. VAE forward pass
2. Node embedding generator
3. GraphTransformer with VAE integration
4. End-to-end training with task loss only
5. Baseline model without VAE
6. Single-node graph handling

All tests should pass with no errors.

## Implementation Details

### Files Modified
- `new_experiments/graph_transformer_lrgb.py` - Main implementation

### Files Added
- `test_vae_embedding_generator.py` - Test suite
- `VAE_EMBEDDING_GENERATOR_README.md` - This file

### Key Features
- Clean, well-documented code
- Handles edge cases (single-node graphs)
- Configurable attention parameters
- Optional VAE refinement via command-line flag
- End-to-end differentiable training

## Expected Behavior

With VAE refinement enabled:
1. Input features are projected to hidden dimension
2. Transformer layers process node embeddings
3. **VAE generator refines embeddings**: For each node i, uses context from all other nodes
4. Graph-level pooling aggregates refined embeddings
5. Output projection produces final predictions

The VAE learns to refine embeddings in a way that improves task performance, without any explicit reconstruction objective.

## References
- Mathematical framework: EP(Ni | N-Ni) for all i
- VAE: Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- Graph Transformers: Attention mechanisms for graph-structured data
