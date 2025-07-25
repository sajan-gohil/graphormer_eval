import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(True)

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
import sys
import shutil
import json
import random
import numpy as np
import datetime
import argparse


os.environ['PYTHONHASHSEED'] = '42'
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

parser = argparse.ArgumentParser(
    description="Graphormer Training Parameters")
parser.add_argument("--edge_type", type=str, default="multi_hop", help="Type of edge encoding (multi_hop, single_hop, etc.)")
parser.add_argument("--enable_spatial_encoder", action="store_true", help="Enable spatial encoder")
parser.add_argument("--enable_diffusion", action="store_true", help="Enable diffusion")
parser.add_argument("--optimize_diffuser", action="store_true", help="Optimize diffuser")
parser.add_argument("--experiment_dir", type=str, default="./experiments", help="Directory to save experiment results")
parser.add_argument("--name", type=str, default="graphormer_experiment", help="Name of the experiment")
parser.add_argument("--reconstruction_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--structure_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--onscreen_logs", action="store_true", help="print logs on screen instead of log files in experiment dir")
parser.add_argument("--batch_size", type=int, default=512, help="number of graphs in a batch")
parser.add_argument("--diffusion_type", type=str, default="x0", help='Type of diffusion predictor ["x0", "delta", "noise_pred"]')
parser.add_argument("--detached_denoiser", action="store_true", help="Detach embedding before passing to diffusion module to separate denoiser training")
parser.add_argument("--pretrained_weights", type=str, default=None, help="path to checkpoint pt file")
parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of diffusion steps for the model")

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

# 1. Dataset setup
split_dict = torch.load("split_dict.pt", weights_only=False)
train_idx = split_dict['train']
valid_idx = split_dict['valid']

# Load preprocessed pyg graph objects
pyg_data = torch.load("pyg_dataset_ogb.pt", weights_only=False)
# # Assign num_nodes attribute
# for i in range(len(pyg_data)):
#     pyg_data[i].num_nodes = pyg_data[i].x.shape[0]
# Create subsets
train_dataset = Subset(pyg_data, train_idx[:len(train_idx) // 10])  # Use a smaller subset for faster training
valid_dataset = Subset(pyg_data, valid_idx)

# Data loaders
BATCH_SIZE = args.batch_size  # 512

collator = GraphormerDataCollator(on_the_fly_processing=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE//8, shuffle=False, collate_fn=collator, num_workers=1)  # Val data has some big samples

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
    **vars(args)
    # edge_type=args.edge_type,
    # enable_spatial_encoder=args.enable_spatial_encoder,
    # enable_diffusion=args.enable_diffusion,
    # optimize_diffuser=args.optimize_diffuser,
    # reconstruction_scale=args.reconstruction_scale,
    # experiment_dir=args.experiment_dir
)

model = GraphormerForGraphClassification(config)
# model.encoder.enable_diffusion = True  # Enable diffusion
if args.pretrained_weights:
    weights = torch.load(args.pretrained_weights, weights_only=True)
    model.load_state_dict(weights)
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

param_list = [{"params": [i for n,i in model.named_parameters() if "diffusion_model" not in n], "lr":LEARNING_RATE}]
if args.enable_diffusion:param_list += [{"params": model.encoder.diffusion_model.parameters(), "lr": 2e-4}]
optimizer = Adam(param_list, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

# diffusion_optimizer = Adam(model.encoder.diffusion_model.parameters(), lr=1e-4)

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
MAX_EPOCHS = 100
best_valid_mae = float('inf')

for epoch in range(MAX_EPOCHS):
    print("EPOCH: ", epoch)
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
    for batch in pbar:
        for k in batch:
            try:
                batch[k] = batch[k].to(device)
            except:
                batch[k] = [i.to(device) for i in batch[k]]
        labels = batch["labels"]

        # outputs = model(**batch)
        # Pass edge_index to model if present
        assert "edge_index" in batch.keys()
        # print("batch index len = ", len(batch["edge_index"]))
        outputs = model(**batch)
        # loss = F.l1_loss(outputs[1].view(-1), labels.view(-1), reduction="mean")
        loss = outputs.loss

        optimizer.zero_grad()
        # diffusion_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        # diffusion_optimizer.step()
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
                try:
                    batch[k] = batch[k].to(device)
                except:
                    batch[k] = [i.to(device) for i in batch[k]]
            labels = batch["labels"]
            outputs = model(**batch)
            y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    valid_mae = evaluator.eval(input_dict)["mae"]
    with open(f"{args.experiment_dir}/val_metric.csv", "a") as f:
        f.write(f"epoch_{epoch},{valid_mae}\n")

    print(f"Validation MAE: {valid_mae:.6f}")
    if valid_mae < best_valid_mae:
        best_valid_mae = valid_mae
        torch.save(model.state_dict(), f"{args.experiment_dir}/training_checkpoints/best_model_{epoch}.pt")
        print("Best model updated.")

    if step >= MAX_STEPS:
        print("Reached max training steps.")
        break

print(f"Best Validation MAE: {best_valid_mae:.6f}")
