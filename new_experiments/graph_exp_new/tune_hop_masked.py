"""Bayesian hyperparameter search (Optuna TPE) for train_hop_masked_light.py.

Each trial runs the training script as a subprocess, reads the per-epoch
validation metric from its output, reports it to Optuna (so poor trials are
pruned early) and returns the best validation value. The training script is
not modified.

Install:  pip install optuna
Run (one worker per GPU; workers share the study through the SQLite file):
  python tune_hop_masked.py --dataset ZINC12k --device cuda:0 --n_trials 40
  python tune_hop_masked.py --dataset ZINC12k --device cuda:1 --n_trials 40
Results:  python tune_hop_masked.py --dataset ZINC12k --show
Extra fixed flags for every trial go after "--" (those flags are then not
searched):
  python tune_hop_masked.py --dataset ZINC12k -- --graph_pool sum --amp
Search space: --space full (default; every training hyperparameter, conditional
ones only when their parent is on) or --space core (width/depth, hops, dropout,
lr, weight decay, batch size, path options, head type). Leave parameters out
with --exclude, e.g. --exclude multihop_attn,mask_type,use_pos_weight.
Not searched: dataset/device/seed/paths, max_epochs/patience (set by the
tuner), --random_mask (ablation), --one_linear (covered by --one_linear_path
and --one_linear_crosshop), --hop_file and the 'single'/'alternating'/'file'
hop modes (they need extra constraints).

Note: build the distance/predecessor caches once before starting parallel
workers (run one short training with the largest max_hops in the search space,
and --use_path_embeddings), so workers don't write the same cache together.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys

import optuna

EPOCH_RE = re.compile(
    r"^epoch\s+(\d+)\s.*\|\s*val loss\s+\S+\s+\S+\s+([-+0-9.eE]+|nan|inf)\s*\|"
    r"\s*test loss\s+\S+\s+\S+\s+([-+0-9.eE]+|nan|inf)")
PARAMS_RE = re.compile(r"trainable params:\s*([0-9.]+)M")


# Parameters searched in --space core (the rest keep the training-script default).
CORE = {"head_dim", "num_heads", "num_global_heads", "num_layers", "ffn_ratio",
        "max_hops", "dropout", "random_global_prob", "lr", "weight_decay",
        "batch_size", "use_path_embeddings", "path_node_terms", "one_linear_path",
        "linear_head"}


def suggest(trial: optuna.Trial, a) -> dict:
    """Search space over the training script's hyperparameters. Returns CLI flags.

    A parameter is searched if it is in the chosen space (--space core|full) and
    not excluded (--exclude, or given as a fixed flag after "--"). Unsearched
    parameters keep the training-script default (or the fixed flag). Conditional
    parameters are only suggested when their parent setting is active.
    """
    on = lambda name: (a.space == "full" or name in CORE) and name not in a.skip
    f = {}

    def cat(name, choices):
        if on(name):
            f[name] = trial.suggest_categorical(name, choices)
        return f.get(name)

    def flag(name):                                       # store_true flags
        if on(name):
            f[name] = trial.suggest_categorical(name, [False, True])
        return f.get(name, False)

    # ---- width / depth (hidden_dim = head_dim * num_heads) ----------------
    head_dim = cat("head_dim", [8, 16, 32]) or 16
    num_heads = cat("num_heads", [4, 6, 8, 12])
    if num_heads is not None:
        f["hidden_dim"] = head_dim * num_heads
    f.pop("head_dim", None)                               # not a training flag
    if on("num_global_heads"):
        f["num_global_heads"] = trial.suggest_int("num_global_heads", 0, 2)
    if on("num_layers"):
        f["num_layers"] = trial.suggest_int("num_layers", 2, 8)
    cat("ffn_ratio", [1.0, 2.0, 4.0])
    cat("norm_type", ["layer", "rms", "graph"])
    flag("block_diag_out")
    flag("final_norm")

    # ---- hop masks ----------------------------------------------------------
    if on("max_hops"):
        f["max_hops"] = trial.suggest_int("max_hops", a.min_hops, a.max_hops)
    mode = cat("hop_mode", ["contiguous", "window", "interleaved"])
    if mode in ("window", "interleaved") and on("hop_window"):
        f["hop_window"] = trial.suggest_int("hop_window", 1, 3)
    if cat("mask_type", ["shortest_path", "adj_power"]) == "adj_power":
        flag("adj_self_loops")
    if on("random_global_prob"):
        f["random_global_prob"] = trial.suggest_categorical(
            "random_global_prob", [0.0, 0.05, 0.1, 0.2])

    # ---- attention variants (multihop and adj-power blending exclude each other)
    if flag("multihop_attn"):
        cat("multihop_readout", ["sum", "mean"])
        flag("multihop_no_global")
    else:
        flag("blend_adj_power")
    if flag("dynamic_cross_hop"):
        flag("one_linear_crosshop")

    # ---- edges ---------------------------------------------------------------
    flag("use_edge_features")
    flag("use_edge_bias")
    if on("num_post_gat_layers"):
        f["num_post_gat_layers"] = trial.suggest_int("num_post_gat_layers", 0, 2)
        if f["num_post_gat_layers"] > 0:
            cat("num_gat_heads", [2, 4])

    # ---- path embeddings ---------------------------------------------------
    if a.path_embeddings == "search":
        use_path = flag("use_path_embeddings")
    else:
        use_path = a.path_embeddings == "on"
        if use_path:
            f["use_path_embeddings"] = True
    if use_path:
        cat("path_node_terms", ["all", "edge", "edge,src", "edge,dst"])
        cat("path_ties", ["single", "uniform"])
        cat("path_norm", ["sum", "term", "none"])
        flag("one_linear_path")

    # ---- output head ---------------------------------------------------------
    cat("graph_pool", ["sum", "mean", "attention"])
    cat("head_norm", ["layer", "none"])
    flag("linear_head")
    if on("head_dropout"):
        f["head_dropout"] = trial.suggest_float("head_dropout", 0.0, 0.5, step=0.1)
    flag("use_pos_weight")

    # ---- optimisation --------------------------------------------------------
    if on("dropout"):
        f["dropout"] = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    if on("lr"):
        f["lr"] = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
    cat("lr_min", [1e-7, 1e-6, 1e-5])
    if on("weight_decay"):
        f["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cat("batch_size", [32, 64, 128, 200])
    cat("grad_clip", [1.0, 5.0, 10.0])
    cat("warmup_ratio", [0.0, 0.01, 0.05])
    if cat("lr_scheduler", ["cosine", "plateau", "constant"]) == "plateau":
        cat("reduce_lr_patience", [5, 10, 20])
    return f


def _check(f: dict, a):
    """Combinations the training script rejects -> prune instead of crashing."""
    heads = f.get("num_heads", a.default_heads)
    hidden = f.get("hidden_dim", a.default_hidden)
    if f.get("num_global_heads", 0) >= heads:
        return "num_global_heads >= num_heads"
    if hidden % heads:
        return "hidden_dim not divisible by num_heads"
    if f.get("num_post_gat_layers", 0) > 0 and hidden % f.get("num_gat_heads", 4):
        return "hidden_dim not divisible by num_gat_heads"
    if f.get("multihop_attn") and f.get("blend_adj_power"):
        return "multihop_attn and blend_adj_power are exclusive"
    return None


def to_cli(flags: dict) -> list:
    out = []
    for k, v in flags.items():
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def objective(trial: optuna.Trial, a, extra: list) -> float:
    flags = suggest(trial, a)
    bad = _check(flags, a)
    if bad:
        raise optuna.TrialPruned(bad)
    if flags.get("lr_scheduler", "cosine") == "cosine" and "cosine_epochs" not in a.skip:
        flags["cosine_epochs"] = a.epochs              # cosine spans the trial length

    save_dir = os.path.join(a.out_dir, f"trial_{trial.number:04d}")
    os.makedirs(save_dir, exist_ok=True)
    cmd = ([sys.executable, a.train_script, "--dataset", a.dataset,
            "--device", a.device, "--seed", str(a.seed),
            "--max_epochs", str(a.epochs), "--patience", str(a.patience),
            "--num_workers", str(a.num_workers), "--save_dir", save_dir]
           + to_cli(flags) + extra)
    trial.set_user_attr("cmd", " ".join(shlex.quote(c) for c in cmd))
    sign = 1.0 if a.direction == "minimize" else -1.0
    best_val, best_test = None, None

    log_path = os.path.join(save_dir, "train.log")
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        try:
            for line in proc.stdout:
                log.write(line)
                m = PARAMS_RE.search(line)
                if m and a.max_params and float(m.group(1)) * 1e6 > a.max_params:
                    trial.set_user_attr("params_M", float(m.group(1)))
                    raise optuna.TrialPruned(f"params {m.group(1)}M > budget")
                if m:
                    trial.set_user_attr("params_M", float(m.group(1)))
                m = EPOCH_RE.search(line)
                if not m:
                    continue
                epoch, val, test = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if val != val:                                   # NaN -> failed trial
                    raise optuna.TrialPruned("NaN validation metric")
                if best_val is None or sign * val < sign * best_val:
                    best_val, best_test = val, test
                trial.report(best_val, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at epoch {epoch}")
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if best_val is not None:
                trial.set_user_attr("best_val", best_val)
                trial.set_user_attr("test_at_best_val", best_test)

    if best_val is None:
        raise RuntimeError(f"trial {trial.number}: no epoch output (see {log_path})")
    return best_val


def show(study: optuna.Study, a):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"{len(study.trials)} trials, {len(done)} complete")
    if not done:
        return
    rev = a.direction == "maximize"
    for t in sorted(done, key=lambda t: t.value, reverse=rev)[:a.top]:
        print(f"\n#{t.number}  val {t.value:.4f}  test {t.user_attrs.get('test_at_best_val')}"
              f"  params {t.user_attrs.get('params_M')}M")
        print("  " + t.user_attrs.get("cmd", ""))
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\nParameter importance (fANOVA):")
        for k, v in imp.items():
            print(f"  {k:<20} {v:.3f}")
    except Exception as e:                                       # needs >= 2 trials
        print(f"(importance not available: {e})")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--direction", choices=["minimize", "maximize"], default=None,
                   help="Default: minimize for MAE datasets, maximize otherwise")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_trials", type=int, default=40, help="Trials for this worker")
    p.add_argument("--timeout", type=float, default=None, help="Seconds for this worker")
    p.add_argument("--epochs", type=int, default=400, help="max_epochs per trial")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min_hops", type=int, default=8)
    p.add_argument("--max_hops", type=int, default=16)
    p.add_argument("--path_embeddings", choices=["on", "off", "search"], default="on")
    p.add_argument("--space", choices=["core", "full"], default="full",
                   help="core: width/depth, hops, dropout, lr, wd, batch, path options, "
                        "head type. full: every training hyperparameter (conditional "
                        "ones only when active).")
    p.add_argument("--exclude", default="",
                   help="Comma list of parameters not to search (training default is "
                        "used), e.g. graph_pool,multihop_attn,use_pos_weight")
    p.add_argument("--default_heads", type=int, default=8,
                   help="Training default num_heads (for checks when not searched)")
    p.add_argument("--default_hidden", type=int, default=128,
                   help="Training default hidden_dim (for checks when not searched)")
    p.add_argument("--max_params", type=float, default=500_000,
                   help="Prune trials above this parameter count (0 = no limit)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="Training seed for every trial")
    p.add_argument("--warmup_epochs", type=int, default=60,
                   help="Epochs before a trial can be pruned")
    p.add_argument("--startup_trials", type=int, default=8,
                   help="Random trials before TPE / pruning start")
    p.add_argument("--study_name", default=None)
    p.add_argument("--storage", default=None, help="Default: sqlite:///optuna_<dataset>.db")
    p.add_argument("--out_dir", default=None, help="Default: optuna_<dataset>/")
    p.add_argument("--train_script", default="train_hop_masked_light.py")
    p.add_argument("--show", action="store_true", help="Print results and exit")
    p.add_argument("--top", type=int, default=5)
    a = p.parse_args(argv)

    # Flags fixed after "--" are not searched (the fixed value is used).
    fixed = {t[2:].split("=")[0] for t in extra if t.startswith("--")}
    a.skip = {x.strip() for x in a.exclude.split(",") if x.strip()} | fixed
    if {"hidden_dim", "num_heads"} & fixed:
        a.skip |= {"head_dim", "num_heads"}
    if a.direction is None:
        a.direction = "minimize" if a.dataset.lower().startswith(("zinc", "peptides-struct")) \
            else "maximize"
    tag = a.dataset.replace("/", "_")
    a.study_name = a.study_name or f"hop_masked_{tag}"
    a.storage = a.storage or f"sqlite:///optuna_{tag}.db"
    a.out_dir = a.out_dir or f"optuna_{tag}"
    os.makedirs(a.out_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=a.study_name, storage=a.storage, direction=a.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True,
                                           n_startup_trials=a.startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=a.startup_trials,
                                           n_warmup_steps=a.warmup_epochs))
    if a.show:
        show(study, a)
        return
    study.optimize(lambda t: objective(t, a, extra), n_trials=a.n_trials,
                   timeout=a.timeout, catch=(RuntimeError,))
    show(study, a)


if __name__ == "__main__":
    main()

