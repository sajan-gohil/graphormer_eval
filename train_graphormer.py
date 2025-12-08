import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# For tensor parallelism
from transformers import enable_full_determinism
from transformers.utils import logging
from transformers.modeling_utils import get_parameter_device
enable_full_determinism(42)

from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score

try:
    from ogb.lsc import PCQM4MEvaluator
except:
    temp = lambda *args: 1
    PCQM4MEvaluator = temp 
from graphormer_hf.modeling_graphormer import GraphormerForNodeClassification
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import GraphormerDataCollator
import dataset_utils

import os
import sys
import shutil
import json
import random
import numpy as np
import datetime
import argparse
from model_utils import load_model, load_optimizer, load_scheduler, save_checkpoint
from forward_pass import forward_pass

import wandb
import dotenv
dotenv.load_dotenv()

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
parser.add_argument("--tensor_parallel", action="store_true", help="Enable tensor parallelism on 2 GPUs (requires >=2 GPUs)")
parser.add_argument("--experiment_dir", type=str, default="./experiments", help="Directory to save experiment results")
parser.add_argument("--name", type=str, default="graphormer_experiment", help="Name of the experiment")
parser.add_argument("--reconstruction_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--structure_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--aug_loss_scale", type=float, default=0, help="How much to weigh augmentation correction's reconstruction loss")
parser.add_argument("--onscreen_logs", action="store_true", help="print logs on screen instead of log files in experiment dir")
parser.add_argument("--batch_size", type=int, default=512, help="number of graphs in a batch")
parser.add_argument("--diffusion_type", type=str, default="x0", help='Type of diffusion predictor ["x0", "delta", "noise_pred"]')
parser.add_argument("--detached_denoiser", action="store_true", help="Detach embedding before passing to diffusion module to separate denoiser training")
parser.add_argument("--pretrained_weights", type=str, default=None, help="path to checkpoint pt file")
parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of diffusion steps for the model")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument("--dataset_name", type=str, default="cora", help="Name of the dataset to use")
parser.add_argument("--create_subgraph", action="store_true", help="Create subgraphs from given large graph")
parser.add_argument("--num_denoiser_layers", type=int, default=3, help="Number of layers in the denoiser: n down, n-1 up + 1 final projection")
parser.add_argument("--denoiser_type", type=str, default="gat", help="Denoiser layer type. Accepted: gat, linear, mha")

parser.add_argument("--optimize_only_diffuser", action="store_true", help="Optimize only the diffuser model")
parser.add_argument("--augment_edges", action="store_true", help="Remove/add dummy edges and calculate separate loss")
parser.add_argument("--gnn_only", action="store_true", help="Instead of diffusion, treat denoiser as gnn")
parser.add_argument("--remove_attn_bias", action="store_true", help="Remove attention bias module altogether")
parser.add_argument("--enable_layerwise_diffusion", action="store_true", help="Perform diffusion after each attention step")
parser.add_argument("--freeze_pretrained_encoder", type=str, default=None, help="Freeze the pretrained encoder and set weights from given path")
parser.add_argument("--freeze_pretrained_diffusion", type=str, default=None, help="Freeze everything till diffusion model and set weights from given path")
parser.add_argument("--mask_random_input_prob", type=float, default=0.0, help="Randomly mask this fraction of input node features during diffusion training")
parser.add_argument("--node_augmentation", action="store_true", help="Perform dummy node addition")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="global learning_rate")
parser.add_argument("--diffusion_lr", type=float, default=2e-5, help="Learning rate for the diffusion model")

parser.add_argument("--debug", action="store_true", help="Enable anomaly detection and verbose logging")
parser.add_argument("--log_memory", action="store_true", help="Log GPU memory usage at various stages")
parser.add_argument("--max_steps", type=int, default=50000, help="Maximum number of training steps")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for learning rate scheduler")

args = parser.parse_args()

if args.debug:
    torch.autograd.set_detect_anomaly(True)
    print("DEBUG MODE: Anomaly detection enabled.")
else:
    torch.autograd.set_detect_anomaly(False)

args.experiment_dir = os.path.join(args.experiment_dir, args.name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(os.path.join(args.experiment_dir, "training_checkpoints"),
            exist_ok=True)

# --- wandb init ---
wandb_name = f"{args.dataset_name}_5391ef"
if args.onscreen_logs:
    wandb_name += "_temp"
wandb.init(
    project=wandb_name,  # Commit hash of last major change
    name="/".join(args.experiment_dir.split("/")[2:]),
    config=vars(args),
    dir=args.experiment_dir,
    # mode="online" if args.onscreen_logs else "offline"
)

if not args.onscreen_logs:
    sys.stdout = open(os.path.join(args.experiment_dir, "training_log.txt"), "w")
    sys.stderr = open(os.path.join(args.experiment_dir, "training_error_log.txt"),"w")

print(f"Experiment directory: {args.experiment_dir}")
print(f"Parameters: {json.dumps(vars(args), indent=4)}")

shutil.copy("graph_diffusion.py", args.experiment_dir)
shutil.copytree("graphormer_hf/", os.path.join(args.experiment_dir, "graphormer_hf"))
shutil.copy("train_graphormer.py", args.experiment_dir)
shutil.copy("dataset_utils.py", args.experiment_dir)

dataset_classes = {
    "cora": 7,
    "citeseer": 6,
    "pubmed": 3,
    "film": 5,
    "deezer": 6,
    "ogbn-arxiv": 40,
    "ogbn-products": 47,
    "pcqm4mv2": 1,  # Regression task
}
# 2. Model Configuration - Graphormer-base
config = GraphormerConfig(
    num_hidden_layers=6,
    embedding_dim=768//4,
    ffn_embedding_dim=768//4,
    num_attention_heads=8,
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    num_classes=dataset_classes[args.dataset_name],  # Default to 1 for regression tasks
    **vars(args)
)

# Data loaders
train_loader, valid_loader, test_loader = dataset_utils.load_data(args.dataset_name, num_workers=args.num_workers, config=config)

model = GraphormerForNodeClassification(config)
# Compile as graph is always same
# if config.diffusion_type != "ddim":
#   # _ = model(torch.randn(1, 2708, 1433))
#   model = torch.compile(model, fullgraph=False, dynamic=True)
#print("Model compiled successfully.")

# --- Log GPU memory after model creation ---
if torch.cuda.is_available() and args.log_memory:
    print(f"[GPU] Memory allocated after model creation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[GPU] Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    wandb.log({"gpu/model_creation_memory_MB": torch.cuda.memory_allocated() / 1024**2,
               "step": 0})


pre_epoch = 0
# Load pretrained weights if specified
if args.optimize_only_diffuser:
    assert args.pretrained_weights is not None, "Pretrained weights must be provided to optimize only the diffuser."

device = torch.device("cuda")
model, pre_epoch = load_model(model, args)
model.to(device)
optimizer = load_optimizer(model, args)
scheduler, reduce_lr_scheduler = load_scheduler(optimizer, args)


LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = 5.0
MAX_STEPS = args.max_steps
MAX_EPOCHS = args.max_steps // len(train_loader) + 1
# Early stopping settings
# Stop if validation micro/macro F1 does not increase AND validation loss does not decrease
# for EARLY_STOP_PATIENCE_EPOCHS consecutive epochs, but only after MIN_STEPS epochs have passed.
EARLY_STOP_PATIENCE_EPOCHS = 2000
MIN_STEPS = 8000
epochs_since_improvement = 0

# 4. Training loop
evaluator = PCQM4MEvaluator()
train_step = 0
val_step = 0
test_step = 0
# Track best validation metrics
best_valid_mae = float('inf') if args.dataset_name in ["pcqm4mv2"] else float("-inf")
best_f1 = -float("inf")
prev_loss = float('-inf')
best_micro_f1 = -float("inf")
best_macro_f1 = -float("inf")
best_val_loss = float('inf')

for epoch in range(pre_epoch, pre_epoch+MAX_EPOCHS):
    print("EPOCH: ", epoch)
    model.train()
    config.current_epoch = epoch
    config.current_step = epoch
    config.current_split = "train"
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
    for batch in pbar:
        assert "edge_index" in batch.keys()
        outputs, labels, node_mask = forward_pass(
            model, batch, device, config, train_loader, "train")

        loss = outputs.loss
        if loss.item() < prev_loss:
            temp_grad_clip = GRAD_CLIP_NORM
            prev_loss = loss.item()

            # Log GPU memory and tensor sizes after forward pass
            if args.log_memory and torch.cuda.is_available():
                wandb.log({"gpu/forward_memory_MB": torch.cuda.memory_allocated() / 1024**2,
                           "step": config.current_step})

        else:
            temp_grad_clip = GRAD_CLIP_NORM  # //2

        optimizer.zero_grad()
        loss.backward()

        # wandb log gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({f"grad_norms/{name}": param.grad.detach().data.norm(2).item(),
                           "step": config.current_step})

        torch.nn.utils.clip_grad_norm_(model.parameters(), temp_grad_clip)
        optimizer.step()
        scheduler.step()
        reduce_lr_scheduler.step(loss.item())
        # wandb log training loss
        wandb.log({"train/loss": loss.item(), "step": config.current_step})

        pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        train_step += 1
        if train_step >= MAX_STEPS:
            break

    # 5. Validation loop
    model.eval()
    config.current_split = "val"
    y_pred, y_true = [], []
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            outputs, labels, node_mask = forward_pass(
                model, batch, device, config, valid_loader, "val")
            # y_pred.append(outputs[1].view(-1).cpu())
            if config.num_classes > 1:
                y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
            else:
                y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())
            val_step += 1

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # Compute validation metrics
    if args.dataset_name in ["pcqm4mv2"]:
        input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
        valid_mae = evaluator.eval(input_dict)["mae"]
        valid_score = str(valid_mae)
        wandb.log({"val/mae": valid_mae, "step": config.current_step})
        # For regression task we treat lower as better; reuse best_valid_mae
        improved = valid_mae < best_valid_mae
        val_loss = valid_mae
    else:
        micro = f1_score(y_true, y_pred, average="micro")
        macro = f1_score(y_true, y_pred, average="macro")
        # classification accuracy
        accuracy = float((y_pred == y_true).to(torch.float32).mean().item())
        valid_score = f'{micro},{macro},{accuracy}'
        valid_mae = micro
        wandb.log({
            "val/micro_f1": micro,
            "val/macro_f1": macro,
            "val/accuracy": accuracy,
            "step": config.current_step
        })
        # For classification, we consider improvement if either micro or macro f1 increases
        improved = (micro > best_micro_f1) or (macro > best_macro_f1) or (
            accuracy > best_f1) or (val_loss < best_val_loss)
        val_loss = float(outputs.loss.detach().cpu().item())
    
    with open(f"{args.experiment_dir}/val_metric.csv", "a") as f:
        f.write(f"epoch_{epoch},{valid_score}\n")

    print(f"Validation MAE: {valid_score}")

    # Early stopping bookkeeping
    # Update best metrics and reset patience counter on improvement
    if not improved:
        epochs_since_improvement += 1
    else:
        epochs_since_improvement = 0
        # Update best trackers
        if args.dataset_name in ["pcqm4mv2"]:
            best_valid_mae = valid_mae
        else:
            # update whichever metric improved
            if micro > best_micro_f1:
                best_micro_f1 = micro
            if macro > best_macro_f1:
                best_macro_f1 = macro
        # update val loss best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Save best model checkpoint
        save_checkpoint(model, optimizer, scheduler, reduce_lr_scheduler, epoch, train_step, args,
                        filename=f"{args.experiment_dir}/training_checkpoints/best_model.pt")
        print("Best model updated.")
        # Prune older best checkpoints, keep only the last 1
        files = os.listdir(os.path.join(args.experiment_dir, "training_checkpoints"))
        files = [os.path.join(args.experiment_dir, "training_checkpoints", i) for i in files if i.startswith("best_model_")]
        # Sort by modification time, newest last
        files_sorted = sorted(files, key=os.path.getmtime)
        to_remove = files_sorted[:-1] if len(files_sorted) > 1 else []
        for f in to_remove:
            os.remove(f)

    # If both the validation score did not improve AND validation loss did not decrease
    # for EARLY_STOP_PATIENCE_EPOCHS, and we've completed at least MIN_STEPS, stop training.
    no_improve_loss = (val_loss >= best_val_loss)
    if epochs_since_improvement >= EARLY_STOP_PATIENCE_EPOCHS and train_step >= MIN_STEPS and no_improve_loss:
        print(f"Early stopping triggered. No improvement for {epochs_since_improvement} epochs and train_step={train_step} >= MIN_STEPS={MIN_STEPS}.")
        break
    # Save latest model checkpoint
    save_checkpoint(model, optimizer, scheduler, reduce_lr_scheduler, epoch, train_step, args,
                    filename=f"{args.experiment_dir}/training_checkpoints/latest_model.pt")

    # Test set results
    # Load best model and get test set results
    if args.dataset_name not in ["pcqm4mv2"] and epoch % 25 == 0:
        config.current_split = "test"
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs, labels, node_mask = forward_pass(
                    model, batch, device, config, test_loader, "test")
                
                if config.num_classes > 1:
                    y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
                else:
                    y_pred.append(outputs[1].view(-1).cpu())
                y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())
                test_step += 1

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        test_accuracy = float((y_pred == y_true).to(torch.float32).mean().item())
        print(f"Test Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Accuracy: {test_accuracy:.4f}")
        with open(f"{args.experiment_dir}/test_metric.csv", "a") as f:
            f.write(f"epoch_{epoch},{micro_f1},{macro_f1},{test_accuracy}\n")
        wandb.log({"test/micro_f1": micro_f1, "test/macro_f1": macro_f1, "test/accuracy": test_accuracy, "step": config.current_step})

    if train_step >= MAX_STEPS:
        print("Reached max training steps.")
        break

print(f"Best Validation MAE: {best_valid_mae:.6f}")


# FINAL Test set results
# Load best model and get test set results
if args.dataset_name not in ["pcqm4mv2"]:
    # Load best model checkpoint
    best_ckpt = sorted(os.listdir(f"{args.experiment_dir}/training_checkpoints"), key=lambda x: os.path.getmtime(os.path.join(args.experiment_dir, "training_checkpoints", x)))[-1]
    state_dicts = torch.load(os.path.join(args.experiment_dir, "training_checkpoints", best_ckpt), map_location=device)
    model.load_state_dict(state_dicts["model"], strict=False)
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs, labels, node_mask = forward_pass(
                model, batch, device, config, test_loader, "test")
            
            if config.num_classes > 1:
                y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
            else:
                y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    best_test_accuracy = float((y_pred == y_true).to(torch.float32).mean().item())
    print(f"BEST Test Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Accuracy: {best_test_accuracy:.4f}")
    with open(f"{args.experiment_dir}/test_metric.csv", "a") as f:
        f.write(f"micro_f1,{micro_f1}\nmacro_f1,{macro_f1}\naccuracy,{best_test_accuracy}\n")
