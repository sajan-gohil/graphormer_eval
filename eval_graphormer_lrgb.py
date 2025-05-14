import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from transformers import AutoModelForGraphClassification, AutoConfig

# Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "graphormer-base-graphclassification"
dataset_name = "PCQM4Mv2"  # One of the LRGB datasets
batch_size = 32

def main():
    # Load LRGB dataset
    dataset = LRGBDataset(root="./data", name=dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load Graphormer model
    config = AutoConfig.from_pretrained(model_name)
    try:
        # Try to load finetuned weights if available
        model = AutoModelForGraphClassification.from_pretrained("finetuned-graphormer-lrgb", config=config)
        print("Loaded finetuned model weights from 'finetuned-graphormer-lrgb'.")
    except Exception:
        # Fallback to base model if finetuned weights not found
        model = AutoModelForGraphClassification.from_pretrained(model_name, config=config)
        print(f"Loaded base model weights from '{model_name}'.")
    model.to(device)
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Graphormer expects 'edge_index', 'x', and 'attn_edge_type' in batch
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                attn_edge_type=getattr(batch, 'attn_edge_type', None),
                batch=batch.batch
            )
            preds = outputs.logits.argmax(dim=-1)
            if hasattr(batch, 'y'):
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
    if total > 0:
        print(f"Accuracy: {correct / total:.4f}")
    else:
        print("No labels found in the dataset for evaluation.")

if __name__ == "__main__":
    main()
