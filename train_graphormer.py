import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
from torch.nn.parameter import UninitializedParameter


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
parser.add_argument("--dataset_name", type=str, default="pcqm4mv2", help="Name of the dataset to use")
parser.add_argument("--create_subgraph", action="store_true", help="Create subgraphs from given large graph")
parser.add_argument("--num_denoiser_layers", type=int, default=4, help="Number of layers in the denoiser")
parser.add_argument("--use_linear_denoiser", action="store_true", help="Use linear layers in the denoiser")
parser.add_argument("--optimize_only_diffuser", action="store_true", help="Optimize only the diffuser model")
parser.add_argument("--augment_edges", action="store_true", help="Remove/add dummy edges and calculate separate loss")
parser.add_argument("--gnn_only", action="store_true", help="Instead of diffusion, treat denoiser as gnn")
parser.add_argument("--remove_attn_bias", action="store_true", help="Remove attention bias module altogether")
parser.add_argument("--enable_layerwise_diffusion", action="store_true", help="Perform diffusion after each attention step")
parser.add_argument("--freeze_pretrained_encoder", type=str, default=None, help="Freeze the pretrained encoder and set weights from given path")
parser.add_argument("--model_parallel", action="store_true", help="Enable manual layer-wise model parallel across two GPUs (not tensor parallel)")
# Distributed training args
parser.add_argument("--distributed", action="store_true", help="Enable multi-GPU DistributedDataParallel training")
parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend (nccl/gloo)")
parser.add_argument("--dist_url", type=str, default="env://", help="URL to set up distributed training (default: env:// for torchrun)")
parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (for large graphs)")
parser.add_argument("--no_ddp_broadcast_buffers", action="store_true", help="Disable DDP buffer broadcasting")

args = parser.parse_args()

#############################################
# Distributed initialization
#############################################
def init_distributed_mode(args):
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return
    if args.dist_url == "env://":
        # torchrun sets these; if missing, fall back to single-process or default localhost rendezvous
        have_master = ("MASTER_ADDR" in os.environ) and ("MASTER_PORT" in os.environ)
        if not have_master:
            # Provide sensible defaults so user running `python train_graphormer.py --distributed` in a notebook doesn't crash
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        # If WORLD_SIZE not set we treat as single-process (no real DDP benefit) unless user explicitly wants multi-GPU.
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", "0"))
        env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
        # If user has multiple GPUs and asked for distributed but world_size==1, warn and downgrade effectively.
        if env_world_size == 1 and torch.cuda.device_count() > 1:
            print("[WARN] Distributed env vars not set via torchrun; proceeding with rank=0 world_size=1. Use: torchrun --nproc_per_node={} train_graphormer.py --distributed ... for multi-GPU.".format(torch.cuda.device_count()))
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
            args.distributed = False
            return
        args.rank = env_rank
        args.world_size = env_world_size
        args.local_rank = env_local_rank
    else:
        # Single-node multi-GPU fallback
        args.rank = 0
        args.world_size = torch.cuda.device_count()
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()

init_distributed_mode(args)

if not args.distributed and int(os.environ.get("WORLD_SIZE", "1")) == 1:
    print("[INFO] Running in single-process mode (no DDP). To enable multi-GPU, launch with torchrun: \n  torchrun --nproc_per_node=<num_gpus> train_graphormer.py --distributed [other args]")

# Create experiment directory only on rank 0
if args.rank == 0:
    args.experiment_dir = os.path.join(args.experiment_dir, args.name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(args.experiment_dir, "training_checkpoints"), exist_ok=True)
else:
    # Temporary placeholder; will receive path from rank 0 via broadcast later
    args.experiment_dir = ""

# Broadcast experiment dir to all ranks
if args.distributed:
    dir_tensor = torch.tensor(len(args.experiment_dir) if args.rank == 0 else 0, dtype=torch.int32, device=f"cuda:{args.local_rank}")
    dist.broadcast(dir_tensor, src=0)
    if args.rank != 0:
        # allocate string
        tmp = torch.empty(dir_tensor.item(), dtype=torch.uint8, device=f"cuda:{args.local_rank}")
    else:
        tmp = torch.tensor(list(bytearray(args.experiment_dir.encode())), dtype=torch.uint8, device=f"cuda:{args.local_rank}")
    dist.broadcast(tmp, src=0)
    if args.rank != 0:
        args.experiment_dir = bytes(tmp.tolist()).decode()

if args.rank == 0 and not args.onscreen_logs:
    sys.stdout = open(os.path.join(args.experiment_dir, "training_log.txt"), "w")
    sys.stderr = open(os.path.join(args.experiment_dir, "training_error_log.txt"),"w")
elif not args.onscreen_logs:
    # Suppress stdout for non-zero ranks unless onscreen requested
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

if args.rank == 0:
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Parameters: {json.dumps(vars(args), indent=4)}")

if args.rank == 0:
    shutil.copy("graph_diffusion.py", args.experiment_dir)
    if not os.path.exists(os.path.join(args.experiment_dir, "graphormer_hf")):
        shutil.copytree("graphormer_hf/", os.path.join(args.experiment_dir, "graphormer_hf"))
    shutil.copy("train_graphormer.py", args.experiment_dir)
if args.distributed:
    dist.barrier()

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

#############################################
# Lazy / uninitialized param materialization BEFORE moving to GPU
#############################################
from torch.nn.parameter import UninitializedParameter

def initialize_lazy_params_cpu(model, loader):
    needs_init = any(isinstance(p, UninitializedParameter) for p in model.parameters())
    if not needs_init:
        return
    if args.rank == 0:
        print("[INIT] Materializing lazy parameters on CPU with a dummy forward pass.")
    model.eval()
    try:
        sample_batch = next(iter(loader))
    except StopIteration:
        return
    with torch.no_grad():
        # Keep on CPU; ensure tensors are tensors (already CPU by default)
        node_mask = getattr(loader.dataset[0], "train_mask", None)
        _ = model(**sample_batch, node_mask=node_mask)
    model.train()
    if args.rank == 0:
        print("[INIT] Lazy parameter materialization complete.")

initialize_lazy_params_cpu(model, train_loader)

# Tensor parallelism (only after materialization; before manual device move)
if getattr(args, "tensor_parallel", False) and not args.distributed:
    assert torch.cuda.device_count() >= 2, "Tensor parallelism requires at least 2 GPUs."
    if infer_auto_device_map is not None and dispatch_model is not None:
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "16GiB" for i in range(torch.cuda.device_count())},
        )
        model = dispatch_model(model, device_map=device_map)
        print(f"Model dispatched across devices: {device_map}")
    else:
        print("Auto device map not inferred; skipping tensor parallel dispatch.")

#############################################
# Dummy forward to initialize any lazy / uninitialized params BEFORE DDP & optimizer
#############################################
def initialize_lazy_params(model, loader, device):
    has_uninit = any(isinstance(p, UninitializedParameter) for p in model.parameters())
    if not has_uninit:
        return
    if args.rank == 0:
        print("[INIT] Found uninitialized parameters. Running dummy forward to materialize shapes.")
    model.eval()
    try:
        sample_batch = next(iter(loader))
    except StopIteration:
        return
    with torch.no_grad():
        for k in sample_batch:
            try:
                sample_batch[k] = sample_batch[k].to(device)
            except:
                sample_batch[k] = [i.to(device) for i in sample_batch[k]]
        node_mask = getattr(loader.dataset[0], "train_mask", None)
        if node_mask is not None:
            node_mask = node_mask.to(device)
        _ = model(**sample_batch, node_mask=node_mask)
    model.train()
    if args.rank == 0:
        print("[INIT] Dummy forward complete.")

# 3. Optimizer and Scheduler (moved after potential initialization)
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 2 # 60000
MAX_STEPS = 1000000
ADAM_EPS = 1e-8
BETA1, BETA2 = 0.9, 0.999
GRAD_CLIP_NORM = 5.0

# Linear warmup and decay scheduler
def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    return max(
        0.0,
        float(MAX_STEPS - current_step) / float(max(1, MAX_STEPS - WARMUP_STEPS))
    )

# Build optimizer BEFORE scheduler
base_model = model.module if isinstance(model, DDP) else model
param_list = [{"params": [p for n,p in base_model.named_parameters() if "diffusion_model" not in n], "lr":LEARNING_RATE}]
if args.enable_diffusion and hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'diffusion_model'):
    param_list += [{"params": base_model.encoder.diffusion_model.parameters(), "lr": 1e-5}]
optimizer = Adam(param_list, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
if args.optimize_only_diffuser:
    assert args.pretrained_weights is not None, "Pretrained weights must be provided to optimize only the diffuser."
    optimizer = Adam(base_model.encoder.diffusion_model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

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


if args.freeze_pretrained_encoder:
    print(f"Freezing pretrained encoder weights from {args.freeze_pretrained_encoder}")
    state_dicts = torch.load(args.freeze_pretrained_encoder, weights_only=False)
    model.load_state_dict(state_dicts["model"], strict=False)
    for name, param in model.named_parameters():
        if "graph_encoder" in name or "GraphEncoder" in name:
            param.requires_grad = False
            print(f"Froze parameter: {name}")

#############################################
# Device move AFTER lazy param init
#############################################
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_parallel:
    assert torch.cuda.device_count() >= 2, "Model parallel requires at least 2 GPUs"
    # Assign layers across first two GPUs
    model.set_model_parallel(["cuda:0", "cuda:1"])  # simple 2-way split
    # Head stays on cuda:0 (handled in set_model_parallel)
else:
    if not (getattr(args, "tensor_parallel", False) and infer_auto_device_map is not None and dispatch_model is not None):
        model.to(device)

# Wrap with DDP
if args.distributed:
    # When using custom model parallel, wrap only rank0? We assume single-process multi-GPU if model_parallel.
    if args.model_parallel and args.world_size > 1:
        if args.rank == 0:
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=not args.no_ddp_broadcast_buffers,
            )
        else:
            # Other ranks build their own copy (still wrapped to keep API uniform) but may OOM; recommend single-process for mp
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=not args.no_ddp_broadcast_buffers,
            )
    else:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=not args.no_ddp_broadcast_buffers,
        )

# Build optimizer AFTER wrapping so param references are correct
base_model = model.module if isinstance(model, DDP) else model
param_list = [{"params": [p for n,p in base_model.named_parameters() if "diffusion_model" not in n], "lr":LEARNING_RATE}]
if args.enable_diffusion and hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'diffusion_model'):
    param_list += [{"params": base_model.encoder.diffusion_model.parameters(), "lr": 1e-5}]
optimizer = Adam(param_list, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
if args.optimize_only_diffuser:
    assert args.pretrained_weights is not None, "Pretrained weights must be provided to optimize only the diffuser."
    optimizer = Adam(base_model.encoder.diffusion_model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

# 4. Training loop
evaluator = PCQM4MEvaluator()
step = 0
MAX_EPOCHS = 2000
best_valid_mae = float('inf') if args.dataset_name in ["pcqm4mv2"] else float("-inf")
best_f1 = -float("inf")
prev_loss = float('-inf')

def ddp_gather_list(data_list, device):
    """Gather a python list (of tensors or numpy arrays convertible to tensors) across ranks and concatenate."""
    if not args.distributed:
        return data_list
    obj = data_list
    gather_list = [None for _ in range(args.world_size)]
    dist.all_gather_object(gather_list, obj)
    merged = []
    for part in gather_list:
        merged.extend(part)
    return merged

for epoch in range(MAX_EPOCHS):
    if args.rank == 0:
        print("EPOCH: ", epoch)
    model.train()
    iterator = train_loader
    if args.rank == 0:
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
    for batch in iterator:
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
        (loss / args.grad_accumulation_steps).backward()
        if (step + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), temp_grad_clip)
            optimizer.step()
            scheduler.step()
            reduce_lr_scheduler.step(loss.item())
            optimizer.zero_grad(set_to_none=True)

        step += 1
        if args.rank == 0:
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        if step >= MAX_STEPS:
            break

    # 5. Validation loop
    model.eval()
    y_pred_local, y_true_local = [], []
    with torch.no_grad():
        for batch in valid_loader:
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    batch[k] = [i.to(device) for i in batch[k]]
            node_mask = getattr(valid_loader.dataset[0], "val_mask", None)
            labels = batch["labels"]
            if node_mask is not None:
                node_mask = node_mask.to(device)
            outputs = model(**batch, node_mask=node_mask)
            logits = outputs[1]
            if config.num_classes > 1:
                y_pred_local.append(torch.argmax(logits, axis=-1).view(-1).cpu())
            else:
                y_pred_local.append(logits.view(-1).cpu())
            y_true_local.append(labels.view(-1).cpu())

    # Gather across ranks
    y_pred_list = ddp_gather_list([t for t in y_pred_local], device)
    y_true_list = ddp_gather_list([t for t in y_true_local], device)
    if args.rank == 0:
        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
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
        # Broadcast score to others for early stopping logic
        metric_tensor = torch.tensor(float(valid_mae) if isinstance(valid_mae, (float,int)) else float(valid_mae), device=device)
    else:
        metric_tensor = torch.zeros(1, device=device)
    if args.distributed:
        dist.broadcast(metric_tensor, src=0)
    current_metric = metric_tensor.item()
    if args.rank == 0:
        is_better = current_metric < best_valid_mae if args.dataset_name in ["pcqm4mv2"] else current_metric >= best_valid_mae
        if is_better:
            best_valid_mae = current_metric
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
    if args.distributed:
        dist.barrier()

    # Test set results
    # Load best model and get test set results
    if args.rank == 0 and args.dataset_name not in ["pcqm4mv2"] and epoch % 25 == 0:
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
    stop_flag = torch.tensor(1 if step >= MAX_STEPS else 0, device=device)
    if args.distributed:
        dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
    if stop_flag.item() > 0:
        if args.rank == 0:
            print("Reached max training steps.")
        break

if args.rank == 0:
    print(f"Best Validation MAE: {best_valid_mae:.6f}")

# Test set results
# Load best model and get test set results
if args.rank == 0 and args.dataset_name not in ["pcqm4mv2"]:
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

if args.distributed:
    dist.barrier()
    dist.destroy_process_group()
