"""
Test for PCGrad gradient computation fix.

This test verifies that the train_epoch function can properly compute gradients
for the auxiliary structure loss when using allow_unused=True and handling None values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'new_experiments'))

from graph_transformer_lrgb_pcgrad import (
    GraphTransformer,
    train_epoch
)


def create_dummy_batch(num_graphs=2, num_nodes_per_graph=5, num_features=10, num_classes=3):
    """Create a dummy batch of graphs for testing."""
    graphs = []
    
    for _ in range(num_graphs):
        # Create node features
        x = torch.randn(num_nodes_per_graph, num_features)
        
        # Create a simple edge index (fully connected for simplicity)
        edge_list = []
        for i in range(num_nodes_per_graph):
            for j in range(i + 1, num_nodes_per_graph):
                edge_list.append([i, j])
                edge_list.append([j, i])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            # If no edges, create a self-loop
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Create target labels (binary classification for each class)
        y = torch.randint(0, 2, (1, num_classes)).float()
        
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


def test_train_epoch_with_struct_loss():
    """Test that train_epoch works with struct_loss_weight > 0."""
    print("Testing train_epoch with struct_loss_weight > 0...")
    
    device = torch.device('cpu')
    
    # Create a small model
    model = GraphTransformer(
        in_dim=10,
        hidden_dim=32,
        out_dim=3,
        layers=2,  # Multiple layers to ensure some parameters are unused by struct_loss
        heads=2,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    batch = create_dummy_batch(num_graphs=2, num_nodes_per_graph=5, num_features=10, num_classes=3)
    
    # Create a simple loader
    class DummyLoader:
        def __init__(self, data):
            self.data = [data]
            self.dataset = data
        
        def __iter__(self):
            return iter(self.data)
    
    loader = DummyLoader(batch)
    
    try:
        # This should NOT raise an error with the fix
        loss, task_loss, attn_loss, struct_loss = train_epoch(
            model, loader, optimizer, device,
            attn_loss_weight=0.0,  # Disable attention loss for simplicity
            struct_loss_weight=0.1  # Enable structure loss
        )
        
        print(f"✓ Train epoch completed successfully!")
        print(f"  Total loss: {loss:.4f}")
        print(f"  Task loss: {task_loss:.4f}")
        print(f"  Attention loss: {attn_loss:.4f}")
        print(f"  Structure loss: {struct_loss:.4f}")
        
        # Verify that struct_loss is non-zero
        assert struct_loss > 0, "Structure loss should be non-zero when enabled"
        print("✓ Structure loss is non-zero as expected")
        
        return True
        
    except RuntimeError as e:
        if "allow_unused" in str(e):
            print(f"✗ Test failed with RuntimeError: {e}")
            print("  The fix was not applied correctly!")
            return False
        else:
            raise


def test_train_epoch_without_struct_loss():
    """Test that train_epoch still works with struct_loss_weight = 0."""
    print("\nTesting train_epoch with struct_loss_weight = 0...")
    
    device = torch.device('cpu')
    
    # Create a small model
    model = GraphTransformer(
        in_dim=10,
        hidden_dim=32,
        out_dim=3,
        layers=2,
        heads=2,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    batch = create_dummy_batch(num_graphs=2, num_nodes_per_graph=5, num_features=10, num_classes=3)
    
    # Create a simple loader
    class DummyLoader:
        def __init__(self, data):
            self.data = [data]
            self.dataset = data
        
        def __iter__(self):
            return iter(self.data)
    
    loader = DummyLoader(batch)
    
    # This should work as before
    loss, task_loss, attn_loss, struct_loss = train_epoch(
        model, loader, optimizer, device,
        attn_loss_weight=0.0,
        struct_loss_weight=0.0  # Disable structure loss
    )
    
    print(f"✓ Train epoch completed successfully!")
    print(f"  Total loss: {loss:.4f}")
    print(f"  Task loss: {task_loss:.4f}")
    
    # Verify that struct_loss is zero
    assert struct_loss == 0, "Structure loss should be zero when disabled"
    print("✓ Structure loss is zero as expected")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PCGrad gradient computation fix")
    print("=" * 60)
    
    # Test with struct_loss enabled (the critical test)
    test1_passed = test_train_epoch_with_struct_loss()
    
    # Test without struct_loss (sanity check)
    test2_passed = test_train_epoch_without_struct_loss()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        print("=" * 60)
        sys.exit(1)
