import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from transformers import GraphormerForGraphClassification, GraphormerModel, AutoConfig
import os

# Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "graphormer-base-graphclassification"
dataset_name = "PCQM4Mv2"  # One of the LRGB datasets
batch_size = 4

# Set task_type manually or infer from dataset name
# Options: 'graph', 'node', 'edge'
task_type = None
if dataset_name.lower() in ["pcqm4mv2"]:
    task_type = "graph"
elif dataset_name.lower() in ["pascalvoc-sp", "coco-sp"]:
    task_type = "node"
elif dataset_name.lower() in ["peptides-func", "peptides-struct"]:
    task_type = "edge"
else:
    raise ValueError(f"Unknown or unsupported dataset: {dataset_name}")

def main():
    # Load LRGB dataset
    dataset = LRGBDataset(root="./data", name=dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load Graphormer model
    config = AutoConfig.from_pretrained(model_name)
    try:
        # Try to load finetuned weights if available
        model = GraphormerForGraphClassification.from_pretrained(f"finetuned-graphormer-lrgb-{dataset_name.lower()}", config=config)
        print(f"Loaded finetuned model weights from 'finetuned-graphormer-lrgb-{dataset_name.lower()}'.")
    except Exception:
        # Fallback to base model if finetuned weights not found
        model = GraphormerForGraphClassification.from_pretrained(model_name, config=config)
        print(f"Loaded base model weights from '{model_name}'.")
    model.to(device)
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                attn_edge_type=getattr(batch, 'attn_edge_type', None),
                batch=batch.batch
            )
            if task_type == "graph":
                preds = outputs.logits.argmax(dim=-1)
                if hasattr(batch, 'y'):
                    correct += (preds == batch.y).sum().item()
                    total += batch.y.size(0)
            elif task_type == "node":
                preds = outputs.logits.argmax(dim=-1)
                if hasattr(batch, 'y'):
                    mask = getattr(batch, 'mask', torch.ones_like(batch.y, dtype=torch.bool))
                    correct += ((preds == batch.y) & mask).sum().item()
                    total += mask.sum().item()
            elif task_type == "edge":
                preds = outputs.logits.argmax(dim=-1)
                if hasattr(batch, 'edge_label'):
                    mask = getattr(batch, 'edge_mask', torch.ones_like(batch.edge_label, dtype=torch.bool))
                    correct += ((preds == batch.edge_label) & mask).sum().item()
                    total += mask.sum().item()
    if total > 0:
        print(f"Accuracy: {correct / total:.4f}")
    else:
        print("No labels found in the dataset for evaluation.")

if __name__ == "__main__":
    main()
