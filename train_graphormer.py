import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(True)

# For tensor parallelism
from transformers import enable_full_determinism
from transformers.utils import logging
from transformers.modeling_utils import get_parameter_device
enable_full_determinism(42)
try:
    from transformers import infer_auto_device_map, dispatch_model
    import torch_xla.core.xla_model as xm
    _ = xm.xla_device()
except ImportError:
    infer_auto_device_map = None
    dispatch_model = None
    is_torch_tpu_available = lambda: False

from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.metrics import f1_score

from ogb.lsc import PCQM4MEvaluator
from graphormer_hf.modeling_graphormer import GraphormerForGraphClassification, GraphormerForNodeClassification
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
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
parser.add_argument("--optimize_diffuser", action="store_true", help="Optimize diffuser  # NOT USED. DEPRECATED")
parser.add_argument("--tensor_parallel", action="store_true", help="Enable tensor parallelism on 2 GPUs (requires >=2 GPUs)")
parser.add_argument("--experiment_dir", type=str, default="./experiments", help="Directory to save experiment results")
parser.add_argument("--name", type=str, default="graphormer_experiment", help="Name of the experiment")
parser.add_argument("--reconstruction_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--structure_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--onscreen_logs", action="store_true", help="print logs on screen instead of log files in experiment dir")
parser.add_argument("--batch_size", type=int, default=512, help="number of graphs in a batch")
parser.add_argument("--diffusion_type", type=str, default="x0", help='Type of diffusion predictor ["x0", "delta", "noise_pred"]')
parser.add_argument("--detached_denoiser", action="store_true", help="Detach embedding before passing to diffusion module to separate denoiser training")
parser.add_argument("--pretrained_weights", type=str, default=None, help="path to checkpoint pt file")
parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of diffusion steps for the model")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument("--dataset_name", type=str, default="pcqm4mv2", help="Name of the dataset to use")
parser.add_argument("--create_subgraph", action="store_true", help="Create subgraphs from given large graph")
parser.add_argument("--num_denoiser_layers", type=int, default=4, help="Number of layers in the denoiser")
parser.add_argument("--use_linear_denoiser", action="store_true", help="Use linear layers in the denoiser")
parser.add_argument("--optimize_only_diffuser", action="store_true", help="Optimize only the diffuser model")
parser.add_argument("--augment_edges", action="store_true", help="Remove/add dummy edges and calculate separate loss")

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

BATCH_SIZE = args.batch_size  # 512
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
    attention_dropout=0.5,
    activation_dropout=0.5,
    num_classes=dataset_classes[args.dataset_name],  # Default to 1 for regression tasks
    **vars(args)
)

# Data loaders
collator = GraphormerDataCollator(on_the_fly_processing=True, config=config)

train_loader, valid_loader, test_loader = dataset_utils.load_data(args.dataset_name, num_workers=args.num_workers, config=config)

if args.dataset_name == "pcqm4mv2":
    model = GraphormerForGraphClassification(config)
else:
    model = GraphormerForNodeClassification(config)
    # Compile as graph is always same
    # if config.diffusion_type != "ddim":model.compile()
    # print("Model compiled successfully.")

# Tensor parallelism: split model across 2 GPUs if requested
if getattr(args, "tensor_parallel", False):
    assert torch.cuda.device_count() >= 2, "Tensor parallelism requires at least 2 GPUs."
    if infer_auto_device_map is not None and dispatch_model is not None:
        # device_map = {k: i % 2 for i, k in enumerate([name for name, _ in model.named_parameters()])}
        if infer_auto_device_map is not None:
            device_map = infer_auto_device_map(
                model,
                max_memory={i: "16GiB" for i in range(torch.cuda.device_count())},
                # no_split_module_classes=["GraphormerBlock", "GraphormerMultiheadAttention"]  # Customize as needed
            )
            model = dispatch_model(model, device_map=device_map)
            print(f"Model dispatched across devices: {device_map}")
        print("Model wrapped for tensor parallelism on GPUs 0 and 1.")
    else:
        print("Tensor parallelism requires transformers >=4.27.0. Proceeding without tensor parallelism.")

# 3. Optimizer and Scheduler
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 2 # 60000
MAX_STEPS = 1000000
ADAM_EPS = 1e-8
BETA1, BETA2 = 0.9, 0.999
GRAD_CLIP_NORM = 5.0

param_list = [{"params": [i for n,i in model.named_parameters() if "diffusion_model" not in n], "lr":LEARNING_RATE}]
if args.enable_diffusion:param_list += [{"params": model.encoder.diffusion_model.parameters(), "lr": 1e-5}]
optimizer = Adam(param_list, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
if args.optimize_only_diffuser:
    assert args.pretrained_weights is not None, "Pretrained weights must be provided to optimize only the diffuser."
    optimizer = Adam(model.encoder.diffusion_model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

# Linear warmup and decay scheduler
def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    return max(
        0.0,
        float(MAX_STEPS - current_step) / float(max(1, MAX_STEPS - WARMUP_STEPS))
    )

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
reduce_lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-8)

# Load pretrained weights if specified
if args.pretrained_weights:
    state_dicts = torch.load(args.pretrained_weights, weights_only=False)
    model.load_state_dict(state_dicts["model"], strict=False)
    optimizer.load_state_dict(state_dicts.get("optimizer", {}))
    if "scheduler" in state_dicts:
        scheduler.load_state_dict(state_dicts["scheduler"])
    if "reduce_lr_scheduler" in state_dicts:
        reduce_lr_scheduler.load_state_dict(state_dicts["reduce_lr_scheduler"])
    print(f"Loaded pretrained weights from {args.pretrained_weights}")


# Only move to device if not tensor parallel (dispatch_model handles device placement)
if not (getattr(args, "tensor_parallel", False) and infer_auto_device_map is not None and dispatch_model is not None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    device = torch.device("cuda:0")

# 4. Training loop
evaluator = PCQM4MEvaluator()
step = 0
MAX_EPOCHS = 2000
best_valid_mae = float('inf')
best_f1 = -float("inf")
prev_loss = float('-inf')

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
        # print(type(train_loader.dataset), dir(train_loader.dataset))
        node_mask = getattr(train_loader.dataset[0], "train_mask", None)
        if args.dataset_name not in ["pcqm4mv2"]:
            assert node_mask is not None
        if node_mask is not None:
            node_mask = node_mask.to(device)
        outputs = model(**batch, node_mask=node_mask)
        # loss = F.l1_loss(outputs[1].view(-1), labels.view(-1), reduction="mean")
        loss = outputs.loss
        if loss.item() < prev_loss:
            temp_grad_clip = GRAD_CLIP_NORM
            prev_loss = loss.item()
        else:
            temp_grad_clip = GRAD_CLIP_NORM  # //2
        optimizer.zero_grad()
        # diffusion_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), temp_grad_clip)
        optimizer.step()
        # diffusion_optimizer.step()
        scheduler.step()
        reduce_lr_scheduler.step(loss.item())

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
            node_mask = getattr(valid_loader.dataset[0], "val_mask", None).view(-1)
            if node_mask is not None:
                node_mask = node_mask.to(device) & ~torch.isnan(labels.view(-1))
            else:
                node_mask = torch.ones(labels.shape, dtype=torch.int32)
            labels = batch["labels"]
            outputs = model(**batch, node_mask=node_mask)
            # y_pred.append(outputs[1].view(-1).cpu())
            if config.num_classes > 1:
                y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
            else:
                y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    
    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    if args.dataset_name in ["pcqm4mv2"]:
        valid_mae = evaluator.eval(input_dict)["mae"]
        valid_score = str(valid_mae)
    else:
        valid_score = f'{f1_score(y_true, y_pred, average="micro")},{f1_score(y_true, y_pred, average="macro")}'
        valid_mae = f1_score(y_true, y_pred, average="micro")
    with open(f"{args.experiment_dir}/val_metric.csv", "a") as f:
        f.write(f"epoch_{epoch},{valid_score}\n")

    print(f"Validation MAE: {valid_score}")
    is_better = valid_mae < best_valid_mae if args.dataset_name in ["pcqm4mv2"] else valid_mae >= best_valid_mae
    if is_better:
        best_valid_mae = valid_mae
        torch.save(
            {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "reduce_lr_scheduler": reduce_lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": step},
            f"{args.experiment_dir}/training_checkpoints/best_model_{epoch}.pt"
        )
        print("Best model updated.")
    torch.save(
            {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "reduce_lr_scheduler": reduce_lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": step},
            f"{args.experiment_dir}/training_checkpoints/latest_model.pt"
        )

    # Test set results
    # Load best model and get test set results
    if args.dataset_name not in ["pcqm4mv2"]:
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                for k in batch:
                    try:
                        batch[k] = batch[k].to(device)
                    except:
                        batch[k] = [i.to(device) for i in batch[k]]
                node_mask = getattr(test_loader.dataset[0], "test_mask", None)
                if node_mask is not None:
                    node_mask = node_mask.to(device)
                labels = batch["labels"]
                outputs = model(**batch, node_mask=node_mask)
                if config.num_classes > 1:
                    y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
                else:
                    y_pred.append(outputs[1].view(-1).cpu())
                y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Test Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
        with open(f"{args.experiment_dir}/test_metric.csv", "a") as f:
            # f.write(f"micro_f1,{micro_f1}\nmacro_f1,{macro_f1}\n")
            f.write(f"epoch_{epoch},{micro_f1},{macro_f1}\n")
    if step >= MAX_STEPS:
        print("Reached max training steps.")
        break

print(f"Best Validation MAE: {best_valid_mae:.6f}")

# Test set results
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
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    batch[k] = [i.to(device) for i in batch[k]]
            node_mask = getattr(test_loader.dataset[0], "test_mask", None)
            if node_mask is not None:
                node_mask = node_mask.to(device)
            labels = batch["labels"]
            outputs = model(**batch, node_mask=node_mask)
            if config.num_classes > 1:
                y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
            else:
                y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
    with open(f"{args.experiment_dir}/test_metric.csv", "a") as f:
        f.write(f"micro_f1,{micro_f1}\nmacro_f1,{macro_f1}\n")