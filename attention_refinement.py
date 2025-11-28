
import torch
import os
import sys
import json
import argparse
import datetime
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from graphormer_hf.modeling_graphormer import GraphormerForGraphClassification, GraphormerForNodeClassification
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import GraphormerDataCollator
import dataset_utils

def main():
    parser = argparse.ArgumentParser(description="Attention Refinement Workflow")
    parser.add_argument("--dataset_name", type=str, default="cora", help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for attention update")
    parser.add_argument("--steps", type=int, default=100, help="Number of refinement steps")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--layer_index", type=int, default=-1, help="Index of the layer to refine (default: last layer)")
    
    # Add other args needed for config
    parser.add_argument("--edge_type", type=str, default="multi_hop")
    parser.add_argument("--enable_spatial_encoder", action="store_true")
    parser.add_argument("--enable_diffusion", action="store_true")
    parser.add_argument("--diffusion_type", type=str, default="x0")
    parser.add_argument("--num_denoiser_layers", type=int, default=3)
    parser.add_argument("--denoiser_type", type=str, default="gat")
    parser.add_argument("--reconstruction_scale", type=float, default=0.0)
    parser.add_argument("--structure_scale", type=float, default=0.0)
    parser.add_argument("--aug_loss_scale", type=float, default=0)
    parser.add_argument("--detached_denoiser", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--augment_edges", action="store_true")
    parser.add_argument("--gnn_only", action="store_true")
    parser.add_argument("--remove_attn_bias", action="store_true")
    parser.add_argument("--enable_layerwise_diffusion", action="store_true")
    parser.add_argument("--freeze_pretrained_encoder", type=str, default=None)
    parser.add_argument("--freeze_pretrained_diffusion", type=str, default=None)
    parser.add_argument("--mask_random_input_prob", type=float, default=0.0)
    parser.add_argument("--experiment_dir", type=str, default="./experiments/attention_refinement")

    args = parser.parse_args()
    
    os.makedirs(args.experiment_dir, exist_ok=True)

    dataset_classes = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
        "film": 5,
        "deezer": 6,
        "ogbn-arxiv": 40,
        "ogbn-products": 47,
        "pcqm4mv2": 1,
    }

    config = GraphormerConfig(
        num_hidden_layers=6,
        embedding_dim=768//4,
        ffn_embedding_dim=768//4,
        num_attention_heads=8,
        dropout=0.0,
        attention_dropout=0.5,
        activation_dropout=0.5,
        num_classes=dataset_classes[args.dataset_name],
        **vars(args)
    )

    print("Loading data...")
    train_loader, valid_loader, test_loader = dataset_utils.load_data(args.dataset_name, num_workers=args.num_workers, config=config)
    
    print("Creating model...")
    if args.dataset_name == "pcqm4mv2":
        model = GraphormerForGraphClassification(config)
    else:
        model = GraphormerForNodeClassification(config)

    if args.pretrained_weights:
        print(f"Loading weights from {args.pretrained_weights}")
        state_dicts = torch.load(args.pretrained_weights, map_location="cpu")
        model.load_state_dict(state_dicts["model"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # We don't want to update model weights, but we need gradients for input/attention

    # Get the target layer
    if args.layer_index < 0:
        layer_idx = config.num_hidden_layers + args.layer_index
    else:
        layer_idx = args.layer_index
    
    print(f"Refining attention for layer {layer_idx}")

    # Process one batch
    for batch in train_loader:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        print(f"\n--- Starting Refinement (Max Steps: {args.steps}, Tolerance: {args.tolerance}) ---")
        
        # Get initial logits
        with torch.no_grad():
            _ = model(**batch)
            target_layer = model.encoder.graph_encoder.layers[layer_idx]
            current_attn_logits = target_layer.self_attn.last_attn_logits.detach().clone()
            current_attn_logits.requires_grad = True

        prev_loss = float('inf')
        initial_loss_val = None
        
        for step in range(args.steps):
            # Forward pass with override
            attn_override = {layer_idx: current_attn_logits}
            
            outputs = model(**batch, attn_override=attn_override)
            loss = outputs.loss
            current_loss = loss.item()
            
            if initial_loss_val is None:
                initial_loss_val = current_loss
            
            diff = abs(current_loss - prev_loss)
            print(f"Step {step}: Loss = {current_loss:.6f}, Diff = {diff:.6f}")
            
            if diff < args.tolerance:
                print(f"Converged at step {step}.")
                break
            
            prev_loss = current_loss
            
            # Backward
            model.zero_grad()
            if current_attn_logits.grad is not None:
                current_attn_logits.grad.zero_()
            
            loss.backward()
            
            # Update attention scores
            with torch.no_grad():
                current_attn_logits -= args.learning_rate * current_attn_logits.grad
                
        print(f"Final Loss: {prev_loss}")
        if initial_loss_val is not None:
            print(f"Total Loss Improvement: {initial_loss_val - prev_loss}")
            
        break # Only process one batch for demonstration

if __name__ == "__main__":
    main()
