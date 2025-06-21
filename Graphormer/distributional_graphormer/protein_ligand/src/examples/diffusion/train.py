import argparse
import logging
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from graphormer.tasks.graph_diffusion import GraphDiffusionTask, GraphDiffusionConfig
from graphormer.models.graphormer_diff import GraphormerDiffModel, GraphormerDiffModelConfig
from graphormer.criterions.diffusion import DiffusionLoss, DiffusionLossConfig
from graphormer.optim.layerwise_adam import LayerwiseAdam, LayerwiseAdamConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="path to dataset directory")
    parser.add_argument("--num-workers", type=int, default=1, help="number of data loading workers")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--max-epoch", type=int, default=1, help="maximum number of epochs")
    parser.add_argument("--warmup-updates", type=int, default=1000, help="number of warmup updates")
    parser.add_argument("--total-num-update", type=int, default=10000, help="total number of updates")
    parser.add_argument("--train-subset", type=str, default="placeholder", help="training subset")
    parser.add_argument("--valid-subset", type=str, default="test_md", help="validation subset")
    parser.add_argument("--validate-interval", type=int, default=1, help="validation interval")
    parser.add_argument("--save-interval", type=int, default=5, help="save interval")
    parser.add_argument("--num-diffusion-timesteps", type=int, default=500, help="number of diffusion timesteps")
    parser.add_argument("--diffusion-sampling", type=str, default="ddpm", help="diffusion sampling strategy")
    parser.add_argument("--ddim-steps", type=int, default=500, help="DDIM steps")
    parser.add_argument("--diffusion-beta-schedule", type=str, default="sigmoid", help="diffusion beta schedule")
    parser.add_argument("--diffusion-layer-add-time-emb", action="store_true", help="add time embedding to node features layerwise")
    parser.add_argument("--diffusion-layer-proj-time-emb", action="store_true", help="project time embedding layerwise")
    parser.add_argument("--diffusion-beta-end", type=float, default=0.02, help="diffusion beta end")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay")
    parser.add_argument("--store-ema", action="store_true", help="store EMA model")
    parser.add_argument("--uses-ema", action="store_true", help="use EMA model")
    parser.add_argument("--ema-fp32", action="store_true", help="use FP32 for EMA model")
    parser.add_argument("--prior-distribution-std", type=float, default=1, help="prior distribution standard deviation")
    parser.add_argument("--pairwise-loss", action="store_true", help="use pairwise loss")
    parser.add_argument("--num-epsilon-estimator", type=int, default=1, help="number of epsilon estimators")
    parser.add_argument("--test-mode", action="store_true", help="test mode")
    parser.add_argument("--finetune-from-model", type=str, help="finetune from model checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
    )

    # Setup task
    task_cfg = GraphDiffusionConfig(
        data_path=args.data_path,
        train_subset=args.train_subset,
        valid_subset=args.valid_subset,
        max_nodes=128,
        uses_ema=args.uses_ema,
    )
    task = GraphDiffusionTask.setup_task(task_cfg)

    # Load datasets
    task.load_dataset(args.train_subset)
    task.load_dataset(args.valid_subset)

    # Setup model
    model_cfg = GraphormerDiffModelConfig(
        num_diffusion_timesteps=args.num_diffusion_timesteps,
        diffusion_timestep_emb_type="positional",
        diffusion_layer_add_time_emb=args.diffusion_layer_add_time_emb,
        diffusion_layer_proj_time_emb=args.diffusion_layer_proj_time_emb,
        diffusion_beta_schedule=args.diffusion_beta_schedule,
        diffusion_beta_end=args.diffusion_beta_end,
        diffusion_sampling=args.diffusion_sampling,
        ddim_steps=args.ddim_steps,
        prior_distribution_std=args.prior_distribution_std,
        pairwise_loss=args.pairwise_loss,
        num_epsilon_estimator=args.num_epsilon_estimator,
        test_mode=args.test_mode,
    )
    model = task.build_model(model_cfg)

    if args.finetune_from_model:
        state_dict = torch.load(args.finetune_from_model, map_location="cpu")
        model.load_state_dict(state_dict["model"])

    if args.local_rank != -1:
        model = model.cuda()
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = model.cuda()

    # Setup criterion
    criterion = DiffusionLoss(task, valid_times=1)

    # Setup optimizer
    optimizer_cfg = LayerwiseAdamConfig(
        adam_betas=(0.9, 0.98),
        adam_eps=1e-8,
        weight_decay=1e-3,
        lr=[0],
        lr_scale_decay=0.65,
    )
    optimizer = LayerwiseAdam(optimizer_cfg, model.parameters())

    # Setup EMA model if needed
    if args.uses_ema:
        ema_model = task.build_model(model_cfg)
        ema_model.load_state_dict(model.state_dict())
        ema_model = ema_model.cuda()
        if args.local_rank != -1:
            ema_model = DistributedDataParallel(ema_model, device_ids=[args.local_rank])
    else:
        ema_model = None

    # Training loop
    for epoch in range(args.max_epoch):
        model.train()
        train_sampler = DistributedSampler(task.datasets[args.train_subset]) if args.local_rank != -1 else None
        train_loader = DataLoader(
            task.datasets[args.train_subset],
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            sampler=train_sampler,
        )

        for batch_idx, sample in enumerate(train_loader):
            sample = {k: v.cuda() for k, v in sample.items()}
            loss, sample_size, logging_output = task.train_step(
                sample, model, criterion, optimizer, batch_idx
            )

            if args.uses_ema:
                with torch.no_grad():
                    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                        ema_param.data.mul_(args.ema_decay).add_(param.data, alpha=1 - args.ema_decay)

            if batch_idx % 100 == 0 and rank == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            if batch_idx % args.validate_interval == 0:
                model.eval()
                valid_sampler = DistributedSampler(task.datasets[args.valid_subset]) if args.local_rank != -1 else None
                valid_loader = DataLoader(
                    task.datasets[args.valid_subset],
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    sampler=valid_sampler,
                )

                valid_loss = 0
                valid_sample_size = 0
                for valid_sample in valid_loader:
                    valid_sample = {k: v.cuda() for k, v in valid_sample.items()}
                    loss, sample_size, logging_output = task.valid_step(
                        valid_sample, model, criterion, ema_model
                    )
                    valid_loss += loss.item() * sample_size
                    valid_sample_size += sample_size

                if rank == 0:
                    logger.info(f"Validation Loss: {valid_loss / valid_sample_size:.4f}")

            if batch_idx % args.save_interval == 0 and rank == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                }
                if args.uses_ema:
                    checkpoint["ema_model"] = ema_model.state_dict()
                torch.save(checkpoint, f"checkpoint_epoch{epoch}_batch{batch_idx}.pt")

if __name__ == "__main__":
    main() 