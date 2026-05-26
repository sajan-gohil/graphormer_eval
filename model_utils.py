import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.optim import Adam, AdamW
from functools import partial
import wandb

ADAM_EPS = 1e-8
BETA1, BETA2 = 0.9, 0.999
WEIGHT_DECAY = 0.0


def log_param_count(module, name):
    """Helper for logging parameter counts"""
    if not module:
        return
    count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in {name}: {count}")
    wandb.log({f"params/{name}": count, "step": 0})


# Linear warmup and decay scheduler
def lr_lambda(current_step, max_steps, warmup_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(max_steps - current_step) / float(max(1, max_steps - warmup_steps))
    )


def log_model_parameters(model):
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

def log_optimizer_parameters(optimizer):
    total_params = 0
    for param_group in optimizer.param_groups:
        group_params = sum(p.numel() for p in param_group['params'] if p.requires_grad)
        total_params += group_params
    print(f"Number of trainable parameters in optimizer: {total_params}")
    wandb.log({"params/optimizer": total_params, "step": 0})


def load_model(model, args):
    # Load pretrained weights if specified
    if args.pretrained_weights:
        state_dicts = torch.load(args.pretrained_weights, weights_only=False)
    elif args.freeze_pretrained_encoder:
        state_dicts = torch.load(args.freeze_pretrained_encoder, weights_only=False)
    elif args.freeze_pretrained_diffusion:
        state_dicts = torch.load(args.freeze_pretrained_diffusion, weights_only=False)
    else:
        return model, 0
    model_state_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in state_dicts["model"].items()
        if k in model_state_dict and v.size() == model_state_dict[k].size()
    }
    # for k,v in pretrained_dict.items():
    #     print("loading:", k)
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    if args.optimize_only_diffuser or args.freeze_pretrained_encoder:
        for param_name, param in model.named_parameters():
            if "graph_encoder" in param_name or "GraphEncoder" in param_name and "diffusion" not in param_name.lower():
                param.requires_grad = False
                param.requires_grad_ = False
                print(f"Froze parameter: {param_name}")

    model.to("cuda")  # TODO: FIX THIS HACK
    return model, state_dicts.get("epoch", 0)


def load_optimizer(model, args):
    param_list = [{
        "params":
        [i for n, i in model.named_parameters() if "diffusion_model" not in n],
        "lr":
        args.learning_rate
    }]
    print("TOTAL PARAMS WITHOUT DIFFUSION = ", len(param_list[0]["params"]))
    # For frozen diffusion, optimizer should not update weights but requires_grad should be true
    if args.enable_diffusion and not args.freeze_pretrained_diffusion:
        param_list += [{
            "params": [i for n, i in model.named_parameters() if "diffusion_model" in n],
            "lr": args.diffusion_lr
        }]
        print("TOTAL PARAMS WITH DIFFUSION = ",
              len(param_list[0]["params"]) + len(param_list[1]["params"]))

    if args.optimize_only_diffuser:
        optimizer = Adam(model.encoder.diffusion_model.parameters(),
                         lr=args.diffusion_lr,
                         betas=(BETA1, BETA2),
                         eps=ADAM_EPS,
                         weight_decay=WEIGHT_DECAY)
    else:
        optimizer = Adam(param_list, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    if args.pretrained_weights:
        state_dicts = torch.load(args.pretrained_weights, weights_only=False)
        try:
            optimizer.load_state_dict(state_dicts.get("optimizer", {}))
        except:
            print("X-X-X-X-X-X-X-X-X\nLOADING OPTIMIZER PRETRAINED FAILED\n######################")

    # Ensure optimizer states are on the same device as model params
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(next(model.parameters()).device)

    return optimizer


def load_scheduler(optimizer, args):
    scheduler = LambdaLR(optimizer, lr_lambda=partial(lr_lambda,
                         max_steps=args.max_steps,
                         warmup_steps=args.warmup_steps))
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=500, min_lr=1e-8)
    if args.pretrained_weights:
        state_dicts = torch.load(args.pretrained_weights, weights_only=False)
        if scheduler and "scheduler" in state_dicts:
            scheduler.load_state_dict(state_dicts["scheduler"])
        if reduce_lr_scheduler and "reduce_lr_scheduler" in state_dicts:
            reduce_lr_scheduler.load_state_dict(state_dicts["reduce_lr_scheduler"])
    return scheduler, reduce_lr_scheduler


def save_checkpoint(model, optimizer, scheduler, reduce_lr_scheduler, epoch, step, args, filename):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "reduce_lr_scheduler": reduce_lr_scheduler.state_dict() if reduce_lr_scheduler else None,
        "epoch": epoch,
        "step": step,
    }
    torch.save(checkpoint, filename)
