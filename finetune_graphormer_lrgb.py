import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from transformers import AutoModelForGraphClassification, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "graphormer-base-graphclassification"
dataset_name = "PCQM4Mv2"
batch_size = 32
epochs = 3
lr = 2e-5

# Load LRGB dataset and split into train/val
dataset = LRGBDataset(root="./data", name=dataset_name)
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Load Graphormer model
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForGraphClassification.from_pretrained(model_name, config=config)
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
            preds = outputs.logits.argmax(dim=-1)
            if hasattr(batch, 'y'):
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
    return correct / total if total > 0 else 0.0

# Training loop
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
        loss = loss_fn(outputs.logits, batch.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {val_acc:.4f}")

# Save the finetuned model
model.save_pretrained("finetuned-graphormer-lrgb")
print("Finetuning complete. Model saved to 'finetuned-graphormer-lrgb'.")
