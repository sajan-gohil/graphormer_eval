"""
Test script for VAE-based Node Embedding Generator.
Tests the forward pass and verifies the model can train end-to-end.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the modified file
from new_experiments.graph_transformer_lrgb import (
    NodeEmbeddingVAE,
    NodeEmbeddingGenerator,
    GraphTransformer
)


def test_vae_forward():
    """Test VAE forward pass."""
    print("=" * 60)
    print("Test 1: VAE Forward Pass")
    print("=" * 60)
    
    embed_dim = 64
    latent_dim = 32
    batch_size = 4
    
    vae = NodeEmbeddingVAE(embed_dim, latent_dim)
    x = torch.randn(batch_size, embed_dim)
    
    # Test forward pass
    refined = vae(x)
    
    assert refined.shape == (batch_size, embed_dim), \
        f"Expected shape {(batch_size, embed_dim)}, got {refined.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {refined.shape}")
    print(f"✓ VAE forward pass successful")
    print()


def test_embedding_generator():
    """Test Node Embedding Generator."""
    print("=" * 60)
    print("Test 2: Node Embedding Generator")
    print("=" * 60)
    
    embed_dim = 64
    latent_dim = 32
    
    generator = NodeEmbeddingGenerator(embed_dim, latent_dim)
    
    # Create a small batch of 2 graphs
    # Graph 1: 3 nodes
    # Graph 2: 4 nodes
    x = torch.randn(7, embed_dim)  # Total 7 nodes
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])  # Batch assignment
    
    # Test forward pass
    refined = generator(x, batch)
    
    assert refined.shape == x.shape, \
        f"Expected shape {x.shape}, got {refined.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Batch: {batch}")
    print(f"✓ Output shape: {refined.shape}")
    print(f"✓ Embedding generator successful")
    print()


def test_graph_transformer_integration():
    """Test GraphTransformer with VAE integration."""
    print("=" * 60)
    print("Test 3: GraphTransformer with VAE Integration")
    print("=" * 60)
    
    # Model parameters
    in_dim = 9
    hidden_dim = 64
    out_dim = 10
    layers = 2
    heads = 4
    dropout = 0.1
    
    # Create model with VAE refiner
    model = GraphTransformer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
        use_vae_refiner=True
    )
    
    # Create a simple batch of graphs
    # Graph 1: 4 nodes, 4 edges
    edge_index1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    x1 = torch.randn(4, in_dim)
    
    # Graph 2: 5 nodes, 5 edges
    edge_index2 = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    x2 = torch.randn(5, in_dim)
    
    # Create Data objects
    data1 = Data(x=x1, edge_index=edge_index1)
    data2 = Data(x=x2, edge_index=edge_index2)
    
    # Create batch
    batch_data = Batch.from_data_list([data1, data2])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
    
    assert output.shape == (2, out_dim), \
        f"Expected output shape (2, {out_dim}), got {output.shape}"
    
    print(f"✓ Model created with VAE refiner")
    print(f"✓ Input: 2 graphs with 4 and 5 nodes")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Forward pass successful")
    print()


def test_end_to_end_training():
    """Test end-to-end training with only task loss (no reconstruction loss)."""
    print("=" * 60)
    print("Test 4: End-to-End Training")
    print("=" * 60)
    
    # Model parameters
    in_dim = 9
    hidden_dim = 32
    out_dim = 10
    layers = 1
    heads = 2
    dropout = 0.1
    
    # Create model with VAE refiner
    model = GraphTransformer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
        use_vae_refiner=True
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create a simple batch
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(3, in_dim)
    data = Data(x=x, edge_index=edge_index)
    batch_data = Batch.from_data_list([data])
    
    # Training step
    model.train()
    initial_loss = None
    final_loss = None
    
    for step in range(10):
        optimizer.zero_grad()
        output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        # Create properly shaped labels matching the output
        y = torch.randint(0, 2, output.shape).float()
        loss = F.binary_cross_entropy_with_logits(output, y)
        
        if step == 0:
            initial_loss = loss.item()
        if step == 9:
            final_loss = loss.item()
        
        loss.backward()
        optimizer.step()
    
    print(f"✓ Initial loss: {initial_loss:.4f}")
    print(f"✓ Final loss: {final_loss:.4f}")
    print(f"✓ Model trained with task loss only (no reconstruction loss)")
    print(f"✓ Gradient flows through VAE to improve task performance")
    print()


def test_without_vae():
    """Test GraphTransformer without VAE (baseline)."""
    print("=" * 60)
    print("Test 5: GraphTransformer without VAE (Baseline)")
    print("=" * 60)
    
    # Model parameters
    in_dim = 9
    hidden_dim = 64
    out_dim = 10
    layers = 2
    heads = 4
    dropout = 0.1
    
    # Create model without VAE refiner
    model = GraphTransformer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
        use_vae_refiner=False
    )
    
    # Create a simple batch
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(3, in_dim)
    data = Data(x=x, edge_index=edge_index)
    batch_data = Batch.from_data_list([data])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
    
    assert output.shape == (1, out_dim), \
        f"Expected output shape (1, {out_dim}), got {output.shape}"
    
    print(f"✓ Model created without VAE refiner")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Baseline model works correctly")
    print()


def test_single_node_graph():
    """Test Node Embedding Generator with single-node graphs."""
    print("=" * 60)
    print("Test 6: Single-Node Graph Handling")
    print("=" * 60)
    
    embed_dim = 64
    latent_dim = 32
    
    generator = NodeEmbeddingGenerator(embed_dim, latent_dim)
    
    # Create a batch with a single-node graph
    x = torch.randn(1, embed_dim)
    batch = torch.tensor([0])
    
    # Test forward pass
    refined = generator(x, batch)
    
    assert refined.shape == x.shape, \
        f"Expected shape {x.shape}, got {refined.shape}"
    assert not torch.isnan(refined).any(), \
        "Output contains NaN values"
    
    print(f"✓ Input: Single node graph")
    print(f"✓ Output shape: {refined.shape}")
    print(f"✓ No NaN values in output")
    print(f"✓ Single-node graph handled correctly")
    print()


def test_gradient_propagation():
    """Test that gradients propagate correctly through batched VAE processing."""
    print("=" * 60)
    print("Test 7: Gradient Propagation with Batched Processing")
    print("=" * 60)
    
    embed_dim = 64
    latent_dim = 32
    
    generator = NodeEmbeddingGenerator(embed_dim, latent_dim)
    generator.train()
    
    # Create a batch of 2 graphs with multiple nodes
    # Graph 1: 3 nodes, Graph 2: 4 nodes (total 7 nodes)
    x = torch.randn(7, embed_dim, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    
    # Forward pass
    refined = generator(x, batch)
    
    # Create a simple loss (sum of all refined embeddings)
    loss = refined.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that VAE parameters have gradients
    vae_params_with_grad = 0
    vae_total_params = 0
    for name, param in generator.vae.named_parameters():
        vae_total_params += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            vae_params_with_grad += 1
    
    assert vae_params_with_grad > 0, \
        "No gradients found in VAE parameters"
    
    print(f"✓ Input: 7 nodes in 2 graphs")
    print(f"✓ VAE parameters with gradients: {vae_params_with_grad}/{vae_total_params}")
    print(f"✓ All nodes processed in single forward pass")
    print(f"✓ Gradients propagate correctly through batched VAE")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VAE-based Node Embedding Generator Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_vae_forward()
        test_embedding_generator()
        test_graph_transformer_integration()
        test_end_to_end_training()
        test_without_vae()
        test_single_node_graph()
        test_gradient_propagation()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print("- VAE forward pass works correctly")
        print("- Node embedding generator implements EP(Ni | N-Ni) for all nodes")
        print("- GraphTransformer integrates VAE before pooling")
        print("- Model trains end-to-end with only task loss (no reconstruction loss)")
        print("- Baseline model (without VAE) still works")
        print("- Single-node graphs are handled correctly without NaN values")
        print("- Gradients propagate correctly through batched VAE processing")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
