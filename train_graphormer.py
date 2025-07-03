import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ogb.lsc import PCQM4Mv2Dataset, PCQM4MEvaluator
from graphormer_hf.modeling_graphormer import GraphormerForGraphClassification
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import GraphormerDataCollator

import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '42'
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# 1. Dataset setup
split_dict = torch.load("split_dict.pt", weights_only=False)
train_idx = split_dict['train']
valid_idx = split_dict['valid']

# Load preprocessed pyg graph objects
pyg_data = torch.load("pyg_dataset_ogb.pt", weights_only=False)

# Assign num_nodes attribute
for i in range(len(pyg_data)):
    pyg_data[i].num_nodes = pyg_data[i].x.shape[0]

# Create subsets
train_dataset = Subset(pyg_data, train_idx)
valid_dataset = Subset(pyg_data, valid_idx)

# Data loaders
BATCH_SIZE = 1024

collator = GraphormerDataCollator(on_the_fly_processing=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE//4, shuffle=False, collate_fn=collator, num_workers=2)  # Val data has some big samples

# 2. Model Configuration - Graphormer-base
config = GraphormerConfig(
    num_hidden_layers=12,
    embedding_dim=768,
    ffn_embedding_dim=768,
    num_attention_heads=32,
    dropout=0.0,
    attention_dropout=0.1,
    activation_dropout=0.1,
    num_classes=1,
    edge_type="single_hop",
)

model = GraphormerForGraphClassification(config)
model.encoder.enable_diffusion = True  # Enable diffusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Optimizer and Scheduler
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 60000
MAX_STEPS = 1000000
ADAM_EPS = 1e-8
BETA1, BETA2 = 0.9, 0.999
GRAD_CLIP_NORM = 5.0

optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

# Linear warmup and decay scheduler
def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    return max(
        0.0,
        float(MAX_STEPS - current_step) / float(max(1, MAX_STEPS - WARMUP_STEPS))
    )

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# 4. Training loop
evaluator = PCQM4MEvaluator()
step = 0
MAX_EPOCHS = 300
best_valid_mae = float('inf')

for epoch in range(MAX_EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
    for batch in pbar:
        for k in batch:
            batch[k] = batch[k].to(device)
        labels = batch["labels"]

        outputs = model(**batch)
        # Pass edge_index to model if present
        if "edge_index" in batch:
            outputs = model(**batch, edge_index=batch["edge_index"])
        else:
            outputs = model(**batch)
        loss = F.l1_loss(outputs[1].view(-1), labels.view(-1), reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        step += 1
        pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        if step >= MAX_STEPS:
            break

    # 5. Validation loop
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in valid_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            labels = batch["labels"]
            outputs = model(**batch)
            # Pass edge_index to model if present
            if "edge_index" in batch:
                outputs = model(**batch, edge_index=batch["edge_index"])
            else:
                outputs = model(**batch)
            y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    valid_mae = evaluator.eval(input_dict)["mae"]

    print(f"Validation MAE: {valid_mae:.6f}")
    if valid_mae < best_valid_mae:
        best_valid_mae = valid_mae
        torch.save(model.state_dict(), f"training_checkpoints/best_model_{epoch}.pt")
        print("Best model updated.")

    if step >= MAX_STEPS:
        print("Reached max training steps.")
        break

print(f"Best Validation MAE: {best_valid_mae:.6f}")
