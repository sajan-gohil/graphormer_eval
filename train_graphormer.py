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

try:
    from ogb.lsc import PCQM4MEvaluator
except:
    temp = lambda *args: 1
    PCQM4MEvaluator = temp 
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
parser.add_argument("--optimize_diffuser", action="store_true", help="Optimize diffuser  # NOT USED. DEPRECATED")
parser.add_argument("--tensor_parallel", action="store_true", help="Enable tensor parallelism on 2 GPUs (requires >=2 GPUs)")
parser.add_argument("--experiment_dir", type=str, default="./experiments", help="Directory to save experiment results")
parser.add_argument("--name", type=str, default="graphormer_experiment", help="Name of the experiment")
parser.add_argument("--reconstruction_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--structure_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
parser.add_argument("--aug_loss_scale", type=float, default=1, help="How much to weigh augmentation correction's reconstruction loss")
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
parser.add_argument("--use_linear_denoiser", action="store_true", help="[Deprecated with denoisers.py] Use linear layers in the denoiser")
parser.add_argument("--denoiser_type", type=str, default="gat", help="Denoiser layer type. Accepted: gat, linear, mha")

parser.add_argument("--optimize_only_diffuser", action="store_true", help="Optimize only the diffuser model")
parser.add_argument("--augment_edges", action="store_true", help="Remove/add dummy edges and calculate separate loss")
parser.add_argument("--gnn_only", action="store_true", help="Instead of diffusion, treat denoiser as gnn")
parser.add_argument("--remove_attn_bias", action="store_true", help="Remove attention bias module altogether")
parser.add_argument("--enable_layerwise_diffusion", action="store_true", help="Perform diffusion after each attention step")
parser.add_argument("--freeze_pretrained_encoder", type=str, default=None, help="Freeze the pretrained encoder and set weights from given path")
parser.add_argument("--freeze_pretrained_diffusion", type=str, default=None, help="Freeze everything till diffusion model and set weights from given path")
parser.add_argument("--mask_random_input_prob", type=float, default=0.0, help="Randomly mask this fraction of input node features during diffusion training")

args = parser.parse_args()

args.experiment_dir = os.path.join(args.experiment_dir, args.name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(os.path.join(args.experiment_dir, "training_checkpoints"),
            exist_ok=True)

# --- wandb init ---
wandb.init(
    project=f"{args.dataset_name}_38bd281" + "_temp" if args.onscreen_logs else f"{args.dataset_name}_38bd281",  # Commit hash of last major change
    name="/".join(args.experiment_dir.split("/")[1:]),
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
    # if config.diffusion_type != "ddim":
    #   # _ = model(torch.randn(1, 2708, 1433))
    #   model = torch.compile(model, fullgraph=False, dynamic=True)
    #print("Model compiled successfully.")

# --- Log GPU memory after model creation ---
if torch.cuda.is_available():
    print(f"[GPU] Memory allocated after model creation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[GPU] Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    wandb.log({"gpu/model_creation_memory_MB": torch.cuda.memory_allocated() / 1024**2}, step=config.current_step)


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
        print("Auto device map not inferred.")

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

pre_epoch = 0
# Load pretrained weights if specified
if args.pretrained_weights:
    state_dicts = torch.load(args.pretrained_weights, weights_only=False)
    #model.load_state_dict(state_dicts["model"], strict=False)
    model_state_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in  state_dicts["model"].items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    for k,v in pretrained_dict.items():
        print("loading:", k)
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.to("cuda")  # TODO: FIX THIS HACK

    try:
        optimizer.load_state_dict(state_dicts.get("optimizer", {}))
    except:
        print("==========================\nLOADING OPTIMIZER PRETRAINED FAILED\n######################################")
    # Ensure optimizer states are on the same device as model params
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(next(model.parameters()).device)

    if "scheduler" in state_dicts:
        scheduler.load_state_dict(state_dicts["scheduler"])
    if "reduce_lr_scheduler" in state_dicts:
        reduce_lr_scheduler.load_state_dict(state_dicts["reduce_lr_scheduler"])
    if "epoch" in state_dicts:
        pre_epoch = state_dicts["epoch"]
    print(f"Loaded pretrained weights from {args.pretrained_weights}")

if args.optimize_only_diffuser:
    assert args.pretrained_weights is not None, "Pretrained weights must be provided to optimize only the diffuser."
    for param_name, param in model.named_parameters():
        if "graph_encoder" in param_name or "GraphEncoder" in param_name and "diffusion" not in param_name.lower():
            param.requires_grad = False
            param.requires_grad_ = False
            print(f"Froze parameter: {param_name}")
    optimizer = Adam(model.encoder.diffusion_model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-8)
    pre_epoch = 0

    # Log parameter counts after freezing
    log_param_count(model, "model_total_post_freeze")
    if hasattr(model, "encoder"):
        log_param_count(model.encoder, "encoder_post_freeze")
        if hasattr(model.encoder, "diffusion_model"):
            log_param_count(model.encoder.diffusion_model, "diffusion_model_post_freeze")


if args.freeze_pretrained_encoder:
    print(f"Freezing pretrained encoder weights from {args.freeze_pretrained_encoder}")
    state_dicts = torch.load(args.freeze_pretrained_encoder, weights_only=False)
    model.load_state_dict(state_dicts["model"], strict=False)
    for name, param in model.named_parameters():
        if "graph_encoder" in name or "GraphEncoder" in name:
            param.requires_grad = False
            print(f"Froze parameter: {name}")

if args.freeze_pretrained_diffusion:
    print(f"Freezing pretrained encoder and diffusion weights from {args.freeze_pretrained_encoder}")
    state_dicts = torch.load(args.freeze_pretrained_encoder, weights_only=False)
    model.load_state_dict(state_dicts["model"], strict=False)
    for name, param in model.named_parameters():
        if "graph_encoder" in name or "GraphEncoder" in name or "diffusion" in name.lower() or "denoiser" in name.lower():
            param.requires_grad = False
            print(f"Froze parameter: {name}")   

# Only move to device if not tensor parallel (dispatch_model handles device placement)
if not (getattr(args, "tensor_parallel", False) and infer_auto_device_map is not None and dispatch_model is not None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    device = torch.device("cuda:0")
    model.to(device)


def log_param_count(module, name):
    """Helper for logging parameter counts"""
    count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in {name}: {count}")
    wandb.log({f"params/{name}": count}, step=config.current_step)

log_param_count(model, "model_total")
if hasattr(model, "encoder"):
    log_param_count(model.encoder, "encoder")
    if hasattr(model.encoder, "graph_encoder"):
        log_param_count(model.encoder.graph_encoder, "graph_encoder")
    if hasattr(model.encoder, "diffusion_model"):
        log_param_count(model.encoder.diffusion_model, "diffusion_model")
        if hasattr(model.encoder.diffusion_model, "denoiser"):
            log_param_count(model.encoder.diffusion_model.denoiser, "denoiser")
if hasattr(model, "classifier"):
    log_param_count(model.classifier, "classifier")

# Log optimizer parameter groups
for idx, group in enumerate(param_list):
    param_count = sum(p.numel() for p in group["params"] if p.requires_grad)
    print(f"Optimizer param group {idx} trainable params: {param_count}")
    wandb.log({f"params/optimizer_group_{idx}": param_count}, step=config.current_step)


# 4. Training loop
evaluator = PCQM4MEvaluator()
train_step = 0
val_step = 0
test_step = 0
MAX_EPOCHS = 3000
best_valid_mae = float('inf') if args.dataset_name in ["pcqm4mv2"] else float("-inf")
best_f1 = -float("inf")
prev_loss = float('-inf')

for epoch in range(pre_epoch, pre_epoch+MAX_EPOCHS):
    print("EPOCH: ", epoch)
    model.train()
    config.current_epoch = epoch
    config.current_step = epoch
    config.current_split = "train"
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
        outputs = model(**batch, node_mask=node_mask, log_step=config.current_step, log_group="train")
        # loss = F.l1_loss(outputs[1].view(-1), labels.view(-1), reduction="mean")
        loss = outputs.loss
        if loss.item() < prev_loss:
            temp_grad_clip = GRAD_CLIP_NORM
            prev_loss = loss.item()

            # Log GPU memory and tensor sizes after forward pass
            if torch.cuda.is_available():
                wandb.log({"gpu/forward_memory_MB": torch.cuda.memory_allocated() / 1024**2}, step=config.current_step)

        else:
            temp_grad_clip = GRAD_CLIP_NORM  # //2
        optimizer.zero_grad()
        # diffusion_optimizer.zero_grad()
        loss.backward()

        # --- wandb log gradients ---
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())}, step=config.current_step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), temp_grad_clip)
        optimizer.step()
        # diffusion_optimizer.step()
        scheduler.step()
        reduce_lr_scheduler.step(loss.item())

        # --- wandb log training loss ---
        wandb.log({"train/loss": loss.item()}, step=config.current_step)

        pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        train_step += 1
        if train_step >= MAX_STEPS:
            break

    # 5. Validation loop
    model.eval()
    config.current_split = "val"
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
            outputs = model(**batch, node_mask=node_mask, log_step=config.current_step, log_group="val")
            # y_pred.append(outputs[1].view(-1).cpu())
            if config.num_classes > 1:
                y_pred.append(torch.argmax(outputs[1], axis=-1).view(-1, 1)[node_mask].view(-1).cpu())
            else:
                y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1, 1)[node_mask].view(-1).cpu())
            val_step += 1

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    if args.dataset_name in ["pcqm4mv2"]:
        valid_mae = evaluator.eval(input_dict)["mae"]
        valid_score = str(valid_mae)
        wandb.log({"val/mae": valid_mae}, step=config.current_step)
    else:
        valid_score = f'{f1_score(y_true, y_pred, average="micro")},{f1_score(y_true, y_pred, average="macro")}'
        valid_mae = f1_score(y_true, y_pred, average="micro")
        wandb.log({
            "val/micro_f1": f1_score(y_true, y_pred, average="micro"),
            "val/macro_f1": f1_score(y_true, y_pred, average="macro")
        }, step=config.current_step)
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
                "step": train_step},
            f"{args.experiment_dir}/training_checkpoints/best_model_{epoch}.pt"
        )
        print("Best model updated.")
    torch.save(
            {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "reduce_lr_scheduler": reduce_lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": train_step},
            f"{args.experiment_dir}/training_checkpoints/latest_model.pt"
        )

    # Test set results
    # Load best model and get test set results
    if args.dataset_name not in ["pcqm4mv2"] and epoch % 25 == 0:
        config.current_split = "test"
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
                outputs = model(**batch, node_mask=node_mask, log_step=config.current_step, log_group="test")
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
        print(f"Test Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
        with open(f"{args.experiment_dir}/test_metric.csv", "a") as f:
            # f.write(f"micro_f1,{micro_f1}\nmacro_f1,{macro_f1}\n")
            f.write(f"epoch_{epoch},{micro_f1},{macro_f1}\n")
        wandb.log({"test/micro_f1": micro_f1, "test/macro_f1": macro_f1}, step=config.current_step)

    if train_step >= MAX_STEPS:
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
    print(f"BEST Test Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
    with open(f"{args.experiment_dir}/test_metric.csv", "a") as f:
        f.write(f"micro_f1,{micro_f1}\nmacro_f1,{macro_f1}\n")
