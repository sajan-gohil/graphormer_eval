import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from transformers import GraphormerForGraphClassification, GraphormerModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import os

# Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "graphormer-base-graphclassification"
batch_size = 4
epochs = 3
lr = 2e-5

# Use DATASET_NAME from environment if set
dataset_name = os.environ.get("DATASET_NAME", "PCQM4Mv2")

# Map dataset_name to correct LRGBDataset name and task type
DATASET_INFO = {
    "PCQM4Mv2": {"name": "PCQM4Mv2", "task": "graph"},
    "PascalVOC-SP": {"name": "PascalVOC-SP", "task": "node"},
    "COCO-SP": {"name": "COCO-SP", "task": "node"},
    "PCQM-Contact": {"name": "PCQM-Contact", "task": "link"},
    "Peptides-func": {"name": "Peptides-func", "task": "graph"},
    "Peptides-struct": {"name": "Peptides-struct", "task": "regression"},
}
if dataset_name not in DATASET_INFO:
    raise ValueError(f"Unknown or unsupported dataset: {dataset_name}")

lrgb_name = DATASET_INFO[dataset_name]["name"]
task_type = DATASET_INFO[dataset_name]["task"]

# Load LRGB dataset and split into train/val
if task_type in ["graph", "node", "regression"]:
    train_dataset = LRGBDataset(root="./data", name=lrgb_name, split="train")
    val_dataset = LRGBDataset(root="./data", name=lrgb_name, split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
else:
    raise NotImplementedError(f"Task type '{task_type}' is not yet supported in this script.")

# Load Graphormer model
config = AutoConfig.from_pretrained(model_name)
model = GraphormerForGraphClassification.from_pretrained(model_name, config=config)
model.to(device)

optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

loss_fn = torch.nn.CrossEntropyLoss()

def evaluate(loader):
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
    return correct / total if total > 0 else 0.0

# Training loop
best_val_loss = float('inf')
plateau_count = 0
plateau_patience = 2  # Stop if val loss doesn't improve for 2 epochs
min_delta = 1e-4      # Minimum change to count as improvement
val_losses = []

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(
            x=batch.x,
            edge_index=batch.edge_index,
            attn_edge_type=getattr(batch, 'attn_edge_type', None),
            batch=batch.batch
        )
        if task_type == "graph":
            loss = loss_fn(outputs.logits, batch.y)
        elif task_type == "node":
            mask = getattr(batch, 'mask', torch.ones_like(batch.y, dtype=torch.bool))
            loss = loss_fn(outputs.logits[mask], batch.y[mask])
        elif task_type == "edge":
            mask = getattr(batch, 'edge_mask', torch.ones_like(batch.edge_label, dtype=torch.bool))
            loss = loss_fn(outputs.logits[mask], batch.edge_label[mask])
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        loss.backward()
        optimizer.step()
        scheduler.step()
    # Validation
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                attn_edge_type=getattr(batch, 'attn_edge_type', None),
                batch=batch.batch
            )
            if task_type == "graph":
                loss = loss_fn(outputs.logits, batch.y)
            elif task_type == "node":
                mask = getattr(batch, 'mask', torch.ones_like(batch.y, dtype=torch.bool))
                loss = loss_fn(outputs.logits[mask], batch.y[mask])
            elif task_type == "edge":
                mask = getattr(batch, 'edge_mask', torch.ones_like(batch.edge_label, dtype=torch.bool))
                loss = loss_fn(outputs.logits[mask], batch.edge_label[mask])
            val_loss += loss.item()
            val_steps += 1
    val_loss = val_loss / max(val_steps, 1)
    val_losses.append(val_loss)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")
    # Save model after every epoch
    epoch_save_dir = f"finetuned-graphormer-lrgb-{dataset_name.lower()}-epoch{epoch+1}"
    os.makedirs(epoch_save_dir, exist_ok=True)
    model.save_pretrained(epoch_save_dir)
    # Early stopping on plateau
    if best_val_loss - val_loss > min_delta:
        best_val_loss = val_loss
        plateau_count = 0
    else:
        plateau_count += 1
    if plateau_count >= plateau_patience:
        print(f"Validation loss plateaued for {plateau_patience} epochs. Stopping early.")
        break
# Save the final model
save_dir = f"finetuned-graphormer-lrgb-{dataset_name.lower()}"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
print(f"Finetuning complete. Model saved to '{save_dir}'.")
