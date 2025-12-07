import shutil
import torch
import os
import sys
import json
import argparse
import datetime
import numpy as np
import wandb
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from graphormer_hf.modeling_graphormer import GraphormerForNodeClassification
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import GraphormerDataCollator
import dataset_utils
import wandb
import dotenv
dotenv.load_dotenv()

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
    parser.add_argument("--name", type=str, default="attention_refinement")
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
    parser.add_argument("--experiment_dir", type=str, default="./experiments/")
    parser.add_argument("--create_subgraph", action="store_true", help="Create subgraphs from given large graph")
    parser.add_argument("--onscreen_logs", action="store_true", help="print logs on screen instead of log files in experiment dir")
    parser.add_argument("--node_augmentation", action="store_true", help="Perform dummy node addition")

    args = parser.parse_args()
    args.experiment_dir = os.path.join(args.experiment_dir, args.name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(args.experiment_dir, "training_checkpoints"),
            exist_ok=True)

    if not args.onscreen_logs:
        sys.stdout = open(os.path.join(args.experiment_dir, "training_log.txt"), "w")
        sys.stderr = open(os.path.join(args.experiment_dir, "training_error_log.txt"),"w")

    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Parameters: {json.dumps(vars(args), indent=4)}")

    shutil.copy("graph_diffusion.py", args.experiment_dir)
    shutil.copytree("graphormer_hf/", os.path.join(args.experiment_dir, "graphormer_hf"))
    shutil.copy("train_graphormer.py", args.experiment_dir)
    shutil.copy("dataset_utils.py", args.experiment_dir)

    wandb.init(
        project=f"attention_refinement_{args.dataset_name}",
        name=f"layer_{args.layer_index}_lr_{args.learning_rate}",
        config=vars(args),
        dir=args.experiment_dir,
    )

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

    def move_to_device(obj, device):
        """Recursively move tensors/lists/tuples/dicts to device."""
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, list):
            return [move_to_device(o, device) for o in obj]
        if isinstance(obj, tuple):
            return tuple(move_to_device(list(obj), device))
        if isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        return obj

    print(f"Refining attention for layer {layer_idx}")

    def get_node_mask(loader, split_name, device, labels):
        if split_name == "train":
            node_mask = getattr(loader.dataset[0], "train_mask", None)
        elif split_name == "val":
            node_mask = getattr(loader.dataset[0], "val_mask", None)
        elif split_name == "test":
            node_mask = getattr(loader.dataset[0], "test_mask", None)
        else:
            node_mask = None
            
        if node_mask is not None:
            if len(node_mask.shape) == 1:
                 node_mask = node_mask.view(-1)
            node_mask = node_mask.to(device)
            # Ensure mask matches labels shape if needed, or handle NaNs
            if labels is not None:
                 # Flatten labels to match mask if mask is flat
                 flat_labels = labels.view(-1)
                 if node_mask.shape[0] == flat_labels.shape[0]:
                     node_mask = node_mask & ~torch.isnan(flat_labels)
        return node_mask

    def compute_metrics(logits, labels, mask, num_classes):
        if logits is None or labels is None:
            return {}
        
        # Apply mask
        if mask is not None:
            # Ensure mask is boolean
            mask = mask.bool()
            # Flatten logits and labels if they are batched but mask is global or if we just want to treat all nodes same
            # Logits: [B, N, C] -> [B*N, C]
            # Labels: [B, N] -> [B*N]
            # Mask: [N] or [B, N] -> [B*N]
            
            if len(logits.shape) == 3:
                B, N, C = logits.shape
                flat_logits = logits.view(-1, C)
                flat_labels = labels.view(-1)
                if len(mask.shape) == 1 and mask.shape[0] == N:
                     # Broadcast mask? Or assume B=1?
                     # If B=1, mask [N] is fine.
                     if B == 1:
                         flat_mask = mask
                     else:
                         flat_mask = mask.repeat(B)
                else:
                     flat_mask = mask.view(-1)
            else:
                flat_logits = logits
                flat_labels = labels
                flat_mask = mask.view(-1)
                
            valid_logits = flat_logits[flat_mask]
            valid_labels = flat_labels[flat_mask]
        else:
            valid_logits = logits
            valid_labels = labels

        if valid_labels.numel() == 0:
            return {"acc": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}

        if num_classes > 1:
            preds = torch.argmax(valid_logits, dim=-1)
        else:
            preds = valid_logits
            
        preds = preds.cpu().numpy()
        targets = valid_labels.cpu().numpy()
        
        acc = (preds == targets).mean()
        micro = f1_score(targets, preds, average="micro")
        macro = f1_score(targets, preds, average="macro")
        
        return {"acc": acc, "micro_f1": micro, "macro_f1": macro}
    # Process one batch from each split
    config.current_step = 0
    for split_name, loader in [("train", train_loader), ("val", valid_loader), ("test", test_loader)]:
        config.current_step += 1
        config.current_split = split_name
        print(f"\n=== Processing {split_name} set ===")
        for batch in loader:
            config.current_step += 1
            # Move batch to device (handles lists/dicts of tensors)
            batch = move_to_device(batch, device)

            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(device)

            node_mask = get_node_mask(loader, split_name, device, labels)

            print(f"\n--- Starting Refinement for {split_name} (Max Steps: {args.steps}, Tolerance: {args.tolerance}) ---")

            # Get initial logits
            with torch.no_grad():
                _ = model(**batch, node_mask=node_mask)
                target_layer = model.encoder.graph_encoder.layers[layer_idx]
                # ensure logits are on the correct device and can receive grads
                current_attn_logits = target_layer.self_attn.last_attn_logits.detach().clone().to(device)
                current_attn_logits.requires_grad_(True)

            prev_loss = float('inf')
            initial_loss_val = None
            initial_metrics = None
            
            for step in range(args.steps):
                config.current_step += 1
                # Forward pass with override
                attn_override = {layer_idx: current_attn_logits}
                
                outputs = model(**batch, node_mask=node_mask, attn_override=attn_override)
                loss = outputs.loss
                current_loss = loss.item()
                
                # Compute metrics
                metrics = compute_metrics(outputs.logits, labels, node_mask, config.num_classes)
                
                if initial_loss_val is None:
                    initial_loss_val = current_loss
                    initial_metrics = metrics
                
                diff = abs(current_loss - prev_loss)
                print(f"Step {step}: Loss = {current_loss:.6f}, Diff = {diff:.6f}, Acc = {metrics['acc']:.4f}, MicroF1 = {metrics['micro_f1']:.4f}")
                
                wandb.log({
                    f"{split_name}/step": step,
                    f"{split_name}/loss": current_loss,
                    f"{split_name}/loss_diff": diff,
                    f"{split_name}/improvement": initial_loss_val - current_loss,
                    f"{split_name}/acc": metrics['acc'],
                    f"{split_name}/micro_f1": metrics['micro_f1'],
                    f"{split_name}/macro_f1": metrics['macro_f1']
                })

                if diff < args.tolerance:
                    print(f"Converged at step {step}.")
                    break
                
                prev_loss = current_loss
                if split_name == "train":
                    # Backward and update only on train split
                    model.zero_grad()
                    if current_attn_logits.grad is not None:
                        current_attn_logits.grad.zero_()

                    loss.backward()

                    # Update attention scores
                    with torch.no_grad():
                        if current_attn_logits.grad is not None:
                            current_attn_logits -= args.learning_rate * current_attn_logits.grad
                    
            print(f"Final Loss ({split_name}): {prev_loss}")
            if initial_loss_val is not None:
                improvement = initial_loss_val - prev_loss
                print(f"Total Loss Improvement ({split_name}): {improvement}")
                wandb.summary[f"{split_name}_final_loss"] = prev_loss
                wandb.summary[f"{split_name}_total_improvement"] = improvement
                wandb.summary[f"{split_name}_final_acc"] = metrics['acc']
                wandb.summary[f"{split_name}_final_micro_f1"] = metrics['micro_f1']
                
            break # Only process one batch per split

if __name__ == "__main__":
    main()

