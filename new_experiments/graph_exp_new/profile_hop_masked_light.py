# profile_hop_masked_light.py
"""One-off profiler for the light hop-masked model (delete after use).

Same CLI as train_hop_masked_light.py, plus:
  --timing_batches N   batches for the loader / compute timing (default 10)
  --profile_steps N    profiled training steps (default 5)
  --prof_dir DIR       trace output (default ./prof_light)

Example:
  python profile_hop_masked_light.py --dataset PascalVOC-SP --max_hops 30 \
      --use_path_embeddings --num_workers 4 --batch_size 16 --amp

Prints (1) loader vs. compute time per batch, (2) time per labelled model part,
(3) top ops by GPU time and by memory. The trace in --prof_dir opens in
TensorBoard (`tensorboard --logdir DIR`) or https://ui.perfetto.dev.
"""
import argparse
import itertools
import time

import torch
from torch.profiler import (ProfilerActivity, profile, record_function,
                            schedule, tensorboard_trace_handler)

import train_hop_masked_light as T

extra = argparse.ArgumentParser(add_help=False)
extra.add_argument("--timing_batches", type=int, default=10)
extra.add_argument("--profile_steps", type=int, default=5)
extra.add_argument("--prof_dir", default="./prof_light")
pargs, _ = extra.parse_known_args()

args = T.parse_args()
torch.manual_seed(args.seed)
train_loader, _, _, _, task, model = T.build_data_task_model(args)
dev = args.device
use_cuda = str(dev).startswith("cuda:2")
sync = torch.cuda.synchronize if use_cuda else (lambda: None)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
model.train()


def train_step(batch):
    with record_function("data_to_device"):
        pyg, dm, nmask, pdata = T._move_batch_to_device(batch, dev, model.max_hops)
        if pdata is not None:
            pyg.path_data = pdata
    with record_function("forward"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and use_cuda):
            logits, _, aux, _ = model(pyg, dm, nmask)
        loss = task.loss(logits.float(), pyg.y) + aux.float()
    with record_function("backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    with record_function("optimizer"):
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()


# ---------------------------------------------------------------------------
# 1. Loader vs. compute time per batch.
# ---------------------------------------------------------------------------
n = min(pargs.timing_batches, len(train_loader) - 2)
it = iter(train_loader)
next(it)                                         # start workers
t0 = time.perf_counter()
cached = [next(it) for _ in range(n)]            # loader only
t_load = (time.perf_counter() - t0) / n

for b in cached[:2]:                             # warm-up (allocator, kernels)
    train_step(b)
sync()
t0 = time.perf_counter()
for b in cached:                                 # compute only (batches in RAM)
    train_step(b)
sync()
t_comp = (time.perf_counter() - t0) / n
del cached, it                                   # stop this iterator's workers

print(f"\n[timing over {n} batches]")
print(f"  loader only : {t_load:.3f} s/batch")
print(f"  compute only: {t_comp:.3f} s/batch  (to device + fwd + bwd + optim)")
bound = "DATA LOADING" if t_load > 0.8 * t_comp else "GPU / model compute"
print(f"  -> likely bound by: {bound}")
if use_cuda:
    print(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

# ---------------------------------------------------------------------------
# 2. Label model parts (forward only) via hooks / wrappers; no model edits.
# ---------------------------------------------------------------------------
labels = {"data_to_device", "forward", "backward", "optimizer"}


def label_module(mod, name):
    stack = []
    labels.add(name)

    def pre(_m, _inp):
        ctx = record_function(name)
        ctx.__enter__()
        stack.append(ctx)

    def post(_m, _inp, _out):
        stack.pop().__exit__(None, None, None)

    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)


def label_method(obj, attr, name):
    fn = getattr(obj, attr)
    labels.add(name)

    def wrapped(*a, **k):
        with record_function(name):
            return fn(*a, **k)

    setattr(obj, attr, wrapped)


label_module(model.encoder, "encoder")
if model.path_embedder is not None:
    label_module(model.path_embedder, "path_embedder")
for i, layer in enumerate(model.layers):
    label_module(layer.attn, f"L{i}.attn")
    label_module(layer.ffn, f"L{i}.ffn")
    if getattr(layer, "use_cross_hop", False):
        label_module(layer.cross_hop, f"L{i}.cross_hop")
if model.post_gat is not None:
    label_module(model.post_gat, "post_gat")
label_module(model.head, "head")
for attr in ("_build_per_head_mask", "_build_adj_power_masks",
             "_build_adj_blend_bias_from_power", "_build_edge_bias",
             "_build_edge_map"):
    label_method(model, attr, attr)

# ---------------------------------------------------------------------------
# 3. torch.profiler over a few full training steps (includes data loading).
# ---------------------------------------------------------------------------
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
wait, warmup = 1, 2
total = wait + warmup + pargs.profile_steps
with profile(activities=acts,
             schedule=schedule(wait=wait, warmup=warmup, active=pargs.profile_steps),
             on_trace_ready=tensorboard_trace_handler(pargs.prof_dir),
             record_shapes=True, profile_memory=True) as prof:
    for batch in itertools.islice(train_loader, total):
        train_step(batch)
        prof.step()

avg = prof.key_averages()
# torch >= 2.4 renamed cuda_* profiler fields to device_*.
dev_name = "device" if hasattr(avg[0], "device_time_total") else "cuda"
time_key = f"{dev_name}_time_total" if use_cuda else "cpu_time_total"
mem_key = f"self_{dev_name}_memory_usage" if use_cuda else "self_cpu_memory_usage"
print(f"\n[labelled parts, {pargs.profile_steps} steps; sorted by {time_key}]")
part_rows = sorted((e for e in avg if e.key in labels),
                   key=lambda e: getattr(e, time_key), reverse=True)
for e in part_rows:
    gpu_ms = getattr(e, f"{dev_name}_time_total", 0) / 1e3
    print(f"  {e.key:<36} cpu {e.cpu_time_total / 1e3:9.1f} ms   "
          f"gpu {gpu_ms:9.1f} ms   calls {e.count}")

print(f"\n[top ops by {time_key}]")
print(avg.table(sort_by=time_key, row_limit=25))
print("\n[top ops by memory]")
print(avg.table(sort_by=mem_key, row_limit=15))
print(f"\ntrace written to {pargs.prof_dir}")

