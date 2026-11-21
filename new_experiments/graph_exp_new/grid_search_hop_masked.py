"""Coarse-to-fine staged grid search for train_hop_masked_light.py.

Coarse stage: hyperparameters are split into small groups. Groups are searched
one after another; each group runs the full grid of its own parameters with
all other parameters held at the current best (the incumbent), the best
configuration of the group becomes the new incumbent, and the next group
starts from it. Runs are short (--coarse_epochs). --rounds repeats the pass.

Fine stage: for the numeric parameters, a grid of neighbours around the
incumbent value (e.g. lr x {0.5, 0.7, 1, 1.4, 2}), again group by group, at
full length (--fine_epochs / --fine_patience).

Selection is by best validation value over epochs (lower is better for MAE).
The incumbent is part of every group's grid, so the result never gets worse.
Each run is cached by its full command (runs/<hash>/result.json); rerunning
the script resumes and only launches missing runs.

  # plan only (groups and run counts)
  python grid_search_hop_masked.py --dataset ZINC12k --dry_run
  # 2 GPUs x 5 jobs, start from a known config, fixed flags after "--"
  python grid_search_hop_masked.py --dataset ZINC12k --gpus 0,1 --jobs_per_gpu 5 \\
      --base "--hidden_dim 64 --num_heads 4 --num_layers 4 --max_hops 10 \\
              --num_global_heads 1 --use_path_embeddings --linear_head --one_linear" \\
      -- --amp
  # only some groups / custom values
  python grid_search_hop_masked.py --dataset ZINC12k --coarse_groups optim,width_depth \\
      --fine_groups optim --grid "max_hops=8,10,12;lr=5e-4,1e-3"

Parameters fixed after "--" or listed in --exclude are not searched. Flags
not in the search space (--amp, --mrr_mode, ...) go after "--".
Not searched: --one_linear (covered by --one_linear_path and
--one_linear_crosshop; --one_linear in --base sets both), --random_mask,
--hop_file and the single/alternating/file hop modes, --cosine_epochs (set to
the run length when lr_scheduler is cosine, unless fixed after "--").

Note: build the distance/predecessor caches once for every max_hops value in
the grid before running many jobs in parallel (one short run each), so jobs
don't write the same cache file at the same time.
"""

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time

EPOCH_RE = re.compile(
    r"^epoch\s+(\d+)\s.*\|\s*val loss\s+\S+\s+(\S+)\s+([-+0-9.eE]+|nan|inf)\s*\|"
    r"\s*test loss\s+\S+\s+\S+\s+([-+0-9.eE]+|nan|inf)")
PARAMS_RE = re.compile(r"trainable params:\s*([0-9.]+)M")

# ---------------------------------------------------------------------------
# Search space. DEFAULTS = training-script defaults, with hidden_dim written as
# head_dim * num_heads. None = flag omitted (training default applies).
# ---------------------------------------------------------------------------
DEFAULTS = {
    # width / depth
    "head_dim": 16, "num_heads": 8, "num_layers": 4, "num_global_heads": 0,
    "ffn_ratio": 4.0, "norm_type": "layer", "block_diag_out": False, "final_norm": False,
    # hop masks
    "max_hops": 40, "hop_mode": "contiguous", "hop_window": 1,
    "mask_type": "shortest_path", "adj_self_loops": False, "random_global_prob": None,
    # attention variants
    "multihop_attn": False, "multihop_readout": "sum", "multihop_no_global": False,
    "blend_adj_power": False, "dynamic_cross_hop": False, "one_linear_crosshop": False,
    # edges
    "use_edge_features": False, "use_edge_bias": False,
    "num_post_gat_layers": 0, "num_gat_heads": 4,
    # path embeddings
    "use_path_embeddings": False, "path_node_terms": "all", "path_ties": "single",
    "path_pool": "sum", "path_attn_heads": 4, "path_norm": "sum", "path_rebuild": "none",
    "one_linear_path": False,
    # head
    "graph_pool": "sum", "head_norm": "layer", "linear_head": False,
    "head_dropout": None, "use_pos_weight": False,
    # optimisation
    "dropout": 0.2, "lr": 1e-3, "lr_min": 1e-6, "weight_decay": 3e-4, "batch_size": 64,
    "grad_clip": 5.0, "warmup_ratio": 0.01, "lr_scheduler": "cosine",
    "reduce_lr_patience": 10,
}
FLOAT_NONE = {"random_global_prob", "head_dropout"}   # default None, value float
TF = [False, True]

# Coarse groups, searched in this order. Conditional parameters only vary when
# their parent is active (see canonical()); duplicates are removed.
COARSE = [
    ("optim", {"lr": [3e-4, 1e-3, 3e-3], "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3]}),
    ("width_depth", {"head_dim": [8, 16, 32], "num_heads": [4, 8],
                     "num_layers": [2, 4, 6, 8]}),
    ("heads_ffn", {"num_global_heads": [0, 1, 2], "ffn_ratio": [1.0, 2.0, 4.0]}),
    ("hops", {"max_hops": [6, 10, 16, 24],
              "hop_mode": ["contiguous", "window", "interleaved"],
              "hop_window": [1, 2, 3]}),
    ("path_inputs", {"use_path_embeddings": [True],      # [False, True] with --path_embeddings search
                     "path_node_terms": ["all", "edge", "edge,src", "edge,dst"],
                     "one_linear_path": TF}),
    ("path_agg", {"path_ties": ["single", "uniform"], "path_pool": ["sum", "attn"],
                  "path_attn_heads": [2, 4]}),
    ("path_norm_rebuild", {"path_norm": ["sum", "term", "none"],
                           "path_rebuild": ["none", "shared", "per_layer"]}),
    ("attn_variant", {"multihop_attn": TF, "multihop_readout": ["sum", "mean"],
                      "multihop_no_global": TF, "blend_adj_power": TF}),
    ("cross_hop", {"dynamic_cross_hop": TF, "one_linear_crosshop": TF}),
    ("mask", {"mask_type": ["shortest_path", "adj_power"], "adj_self_loops": TF,
              "random_global_prob": [0.0, 0.05, 0.1, 0.2]}),
    ("edges", {"use_edge_features": TF, "use_edge_bias": TF,
               "num_post_gat_layers": [0, 1, 2], "num_gat_heads": [2, 4]}),
    ("norm", {"norm_type": ["layer", "rms", "graph"], "block_diag_out": TF,
              "final_norm": TF}),
    ("head", {"graph_pool": ["sum", "mean", "attention"], "head_norm": ["layer", "none"],
              "linear_head": TF, "use_pos_weight": TF}),   # use_pos_weight: maximize only
    ("regularisation", {"dropout": [0.0, 0.1, 0.2, 0.3],
                        "head_dropout": [None, 0.0, 0.2, 0.4]}),
    ("batch", {"batch_size": [32, 64, 128, 256, 512], "grad_clip": [1.0, 5.0, 10.0]}),
    ("schedule", {"lr_scheduler": ["cosine", "plateau", "constant"],
                  "reduce_lr_patience": [10, 20], "warmup_ratio": [0.0, 0.01, 0.05]}),
]


def _mul(*fs, integer=False, lo=None):
    def f(v):
        if v is None:
            return [None]
        out = [v * x for x in fs]
        if integer:
            out = [int(round(x)) for x in out]
        return [x for x in out if lo is None or x >= lo]
    return f


def _add(*ds, lo=None, hi=None, integer=False):
    def f(v):
        if v is None:
            return [None]
        out = [v + d for d in ds]
        if lo is not None:
            out = [x for x in out if x >= lo]
        if hi is not None:
            out = [x for x in out if x <= hi]
        return [int(x) for x in out] if integer else out
    return f


# Fine groups: neighbours of the incumbent value (None stays None).
FINE = [
    ("optim", {"lr": _mul(0.5, 0.7, 1, 1.4, 2), "weight_decay": _mul(0.3, 1, 3)}),
    ("width_depth", {"head_dim": _mul(0.75, 1, 1.25, integer=True, lo=2),
                     "num_layers": _add(-1, 0, 1, lo=1, integer=True)}),
    ("hops", {"max_hops": _add(-2, -1, 0, 1, 2, lo=1, integer=True)}),
    ("ffn", {"ffn_ratio": _add(-1, 0, 1, lo=0.5)}),
    ("regularisation", {"dropout": _add(-0.1, -0.05, 0, 0.05, 0.1, lo=0, hi=0.6),
                        "head_dropout": _add(-0.1, 0, 0.1, lo=0, hi=0.6)}),
    ("mask", {"random_global_prob": _add(-0.05, -0.025, 0, 0.025, 0.05, lo=0, hi=1)}),
    ("batch", {"batch_size": _mul(0.5, 1, 2, integer=True, lo=1),
               "grad_clip": _mul(0.5, 1, 2)}),
    ("schedule", {"warmup_ratio": _add(-0.01, 0, 0.01, lo=0),
                  "lr_min": _mul(0.1, 1, 10)}),
]

PATH_KEYS = {"path_node_terms", "path_ties", "path_pool", "path_attn_heads",
             "path_norm", "path_rebuild", "one_linear_path"}


def _norm_num(v):
    """Round floats to 3 significant digits so equal values hash equally."""
    if isinstance(v, float):
        return float(f"{v:.3g}")
    return v


def canonical(c: dict) -> dict:
    """Reset inactive conditional parameters to their default (so configs that
    differ only in unused settings collapse) and round floats."""
    c = {k: _norm_num(v) for k, v in c.items()}

    def reset(*keys):
        for k in keys:
            if k in c:
                c[k] = DEFAULTS[k]

    if c.get("hop_mode") == "contiguous":
        reset("hop_window")
    if c.get("multihop_attn"):
        c["blend_adj_power"] = False
    else:
        reset("multihop_readout", "multihop_no_global")
    if c.get("mask_type") != "adj_power" and not c.get("blend_adj_power"):
        reset("adj_self_loops")
    if not c.get("dynamic_cross_hop"):
        reset("one_linear_crosshop")
    if not c.get("num_post_gat_layers"):
        reset("num_gat_heads")
    if not c.get("use_path_embeddings"):
        reset(*PATH_KEYS)
    elif c.get("path_pool") != "attn":
        reset("path_attn_heads")
    if c.get("lr_scheduler") != "plateau":
        reset("reduce_lr_patience")
    return c


def invalid(c: dict, a) -> str:
    """Combinations the training script rejects (skipped, not run)."""
    heads = a.fixed_heads or c.get("num_heads", DEFAULTS["num_heads"])
    hidden = a.fixed_hidden or c.get("head_dim", DEFAULTS["head_dim"]) * heads
    if c.get("num_global_heads", 0) >= heads:
        return "num_global_heads >= num_heads"
    if hidden % heads:
        return "hidden_dim % num_heads"
    if c.get("num_post_gat_layers", 0) > 0 and hidden % c.get("num_gat_heads", 4):
        return "hidden_dim % num_gat_heads"
    if c.get("use_path_embeddings") and c.get("path_pool") == "attn" \
            and hidden % c.get("path_attn_heads", 4):
        return "hidden_dim % path_attn_heads"
    return ""


def key_of(c: dict) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in c.items()))


# ---------------------------------------------------------------------------
# Command line / runs
# ---------------------------------------------------------------------------
def to_cli(c: dict, a, epochs: int, patience: int) -> list:
    out = []
    for k in sorted(c):
        v = c[k]
        if k == "head_dim":
            continue
        if k == "num_heads":
            out += ["--hidden_dim", str(c["head_dim"] * v), "--num_heads", str(v)]
            continue
        if v is None or v is False:
            continue
        out += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    out += ["--max_epochs", str(epochs), "--patience", str(patience), "--seed", str(a.seed)]
    if c.get("lr_scheduler", "cosine") == "cosine" and "cosine_epochs" not in a.fixed:
        out += ["--cosine_epochs", str(epochs)]
    return out + a.extra


def run_one(c: dict, stage: str, a, slots: "queue.Queue") -> dict:
    epochs, patience = ((a.coarse_epochs, a.coarse_patience) if stage == "coarse"
                        else (a.fine_epochs, a.fine_patience))
    args = to_cli(c, a, epochs, patience)
    rid = hashlib.sha1(json.dumps([a.dataset, a.train_script] + args).encode()).hexdigest()[:12]
    d = os.path.join(a.out_dir, "runs", rid)
    res_path = os.path.join(d, "result.json")
    if os.path.exists(res_path):
        res = json.load(open(res_path))
        if not (a.retry_failed and res["status"] == "failed"):
            return res
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, a.train_script, "--dataset", a.dataset,
           "--num_workers", str(a.num_workers), "--save_dir", d]
    res = {"id": rid, "stage": stage, "cfg": c, "status": "failed", "best_val": None,
           "test_at_best": None, "best_epoch": None, "last_epoch": None,
           "params_M": None, "note": ""}
    dev = slots.get()
    try:
        full = cmd[:2] + ["--device", dev] + cmd[2:] + args
        res["cmd"] = " ".join(shlex.quote(x) for x in full)
        open(os.path.join(d, "cmd.txt"), "w").write(res["cmd"] + "\n")
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
        sign = 1.0 if a.direction == "minimize" else -1.0
        with open(os.path.join(d, "train.log"), "w") as log:
            proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
            try:
                for line in proc.stdout:
                    log.write(line)
                    m = PARAMS_RE.search(line)
                    if m:
                        res["params_M"] = float(m.group(1))
                        if a.max_params and res["params_M"] * 1e6 > a.max_params:
                            res["status"], res["note"] = "over_budget", f"{m.group(1)}M params"
                            break
                        continue
                    m = EPOCH_RE.search(line)
                    if not m:
                        continue
                    ep, val, test = int(m.group(1)), float(m.group(3)), float(m.group(4))
                    res["last_epoch"] = ep
                    if val != val:
                        res["note"] = f"NaN val at epoch {ep}"
                        break
                    if res["best_val"] is None or sign * val < sign * res["best_val"]:
                        res.update(best_val=val, test_at_best=test, best_epoch=ep)
                if not res["note"]:                 # stdout closed: training finished
                    proc.wait()
            finally:
                if proc.poll() is None:             # stopped early (budget / NaN / interrupt)
                    proc.kill()
                    proc.wait()
        if not res["note"]:
            if proc.returncode == 0 and res["best_val"] is not None:
                res["status"] = "done"
            else:
                res["note"] = f"exit code {proc.returncode}"
    finally:
        slots.put(dev)
    json.dump(res, open(res_path, "w"), indent=1)
    return res


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
def group_grid(space: dict, inc: dict, a, fine: bool) -> list:
    """Candidate configs of one group around the incumbent (incl. the incumbent)."""
    names = [k for k in space if k in inc]
    if not names:
        return []
    def center(k):
        # None = training default, which is dropout/2 (random_global_prob) or
        # dropout (head_dropout); refine around that value.
        if inc[k] is None and k in FLOAT_NONE and inc.get("dropout") is not None:
            return inc["dropout"] / 2 if k == "random_global_prob" else inc["dropout"]
        return inc[k]
    def fine_values(k):
        c = center(k)
        vs = {_norm_num(x) for x in space[k](c)}
        if inc[k] is None:               # the incumbent (None) already is the centre
            vs.discard(_norm_num(c))
        return sorted(vs, key=repr)
    values = [fine_values(k) if fine else space[k] for k in names]
    seen, out = set(), []
    for combo in itertools.chain([tuple(inc[k] for k in names)], itertools.product(*values)):
        c = canonical({**inc, **dict(zip(names, combo))})
        if invalid(c, a):
            continue
        k = key_of(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def run_group(name: str, cands: list, stage: str, a, slots, pool) -> dict:
    futs = [pool.submit(run_one, c, stage, a, slots) for c in cands]
    results = [f.result() for f in futs]
    ok = [r for r in results if r["status"] == "done"]
    rev = a.direction == "maximize"
    ok.sort(key=lambda r: r["best_val"], reverse=rev)
    print(f"\n[{stage}] group '{name}': {len(cands)} configs, {len(ok)} done", flush=True)
    varying = [k for k in cands[0] if len({repr(c[k]) for c in cands}) > 1]
    for r in ok[:a.show_top]:
        diff = ", ".join(f"{k}={r['cfg'][k]}" for k in varying)
        print(f"  val {r['best_val']:.4f}  test {r['test_at_best']:.4f}  ep {r['best_epoch']:>4}"
              f"  {r['params_M']}M  {diff}", flush=True)
    for r in results:
        if r["status"] != "done":
            diff = ", ".join(f"{k}={r['cfg'][k]}" for k in varying)
            print(f"  {r['status']:<11} {r['note']}  {diff}  ({r['id']})", flush=True)
    with open(os.path.join(a.out_dir, "progress.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.strftime("%F %T"), "stage": stage, "group": name,
                            "n": len(cands), "best": ok[0] if ok else None}) + "\n")
    return ok[0] if ok else None


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def _cast(name: str, s):
    d = DEFAULTS[name]
    if s is True or isinstance(d, bool):
        return s if isinstance(s, bool) else str(s).lower() in ("1", "true", "yes")
    if str(s).lower() == "none":
        return None
    if name in FLOAT_NONE or isinstance(d, float):
        return float(s)
    if isinstance(d, int):
        return int(s)
    return str(s)


def parse_flags(text: str) -> dict:
    toks, out, i = shlex.split(text or ""), {}, 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            raise SystemExit(f"--base: unexpected token {t!r}")
        k = t[2:]
        if "=" in k:
            k, v = k.split("=", 1)
            i += 1
        elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            v = toks[i + 1]
            i += 2
        else:
            v = True
            i += 1
        out[k] = v
    return out


def build_base(a) -> dict:
    raw = parse_flags(a.base)
    inc = dict(DEFAULTS)
    if raw.pop("one_linear", False):
        inc["one_linear_path"] = inc["one_linear_crosshop"] = True
    hidden = raw.pop("hidden_dim", None)
    for k, v in raw.items():
        if k not in DEFAULTS:
            raise SystemExit(f"--base: {k!r} is not a searched parameter; pass it after '--'")
        inc[k] = _cast(k, v)
    if hidden is not None:
        h = int(hidden)
        if h % inc["num_heads"]:
            raise SystemExit("--base: hidden_dim not divisible by num_heads")
        inc["head_dim"] = h // inc["num_heads"]
    if a.path_embeddings == "on":
        inc["use_path_embeddings"] = True
    elif a.path_embeddings == "off":
        inc["use_path_embeddings"] = False
    for k in a.noemit:                   # fixed after "--": the fixed flag is used
        inc.pop(k, None)
    return canonical(inc)


def build_spaces(a):
    overrides = {}
    for part in filter(None, (p.strip() for p in a.grid.split(";"))):
        k, vs = part.split("=", 1)
        k = k.strip()
        if k not in DEFAULTS:
            raise SystemExit(f"--grid: unknown parameter {k!r}")
        overrides[k] = [_cast(k, v.strip()) for v in vs.split(",")]

    def keep(k):
        if k in a.skip:
            return False
        if k in PATH_KEYS and a.path_embeddings == "off":
            return False
        if k == "use_pos_weight" and a.direction == "minimize":
            return False
        return True

    coarse = []
    for name, sp in COARSE:
        sp = {k: overrides.get(k, v) for k, v in sp.items() if keep(k)}
        if "use_path_embeddings" in sp and a.path_embeddings == "search":
            sp["use_path_embeddings"] = TF
        elif "use_path_embeddings" in sp and "use_path_embeddings" not in overrides:
            sp.pop("use_path_embeddings")
        if sp:
            coarse.append((name, sp))
    fine = [(n, {k: f for k, f in sp.items() if keep(k)}) for n, sp in FINE]
    fine = [(n, sp) for n, sp in fine if sp]
    sel = lambda groups, s: groups if s == "all" else \
        [g for g in groups if g[0] in set(s.split(","))] if s != "none" else []
    return sel(coarse, a.coarse_groups), sel(fine, a.fine_groups)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    # argparse rejects a value starting with "--" ("--base --use_path_embeddings");
    # join it as "--base=<value>".
    for i in range(len(argv) - 1):
        if argv[i] == "--base":
            argv[i:i + 2] = [f"--base={argv[i + 1]}"]
            break
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--direction", choices=["minimize", "maximize"], default=None,
                   help="Default: minimize for ZINC / Peptides-struct, else maximize")
    p.add_argument("--base", default="",
                   help="Starting incumbent as training flags (searched, unlike flags after '--')")
    p.add_argument("--path_embeddings", choices=["on", "off", "search"], default="on")
    p.add_argument("--coarse_groups", default="all", help="all | none | comma list")
    p.add_argument("--fine_groups", default="all", help="all | none | comma list")
    p.add_argument("--rounds", type=int, default=1, help="Passes over the coarse groups")
    p.add_argument("--grid", default="",
                   help="Override coarse values: 'lr=3e-4,1e-3;max_hops=8,10,12'")
    p.add_argument("--exclude", default="", help="Comma list of parameters not to search")
    p.add_argument("--coarse_epochs", type=int, default=300)
    p.add_argument("--coarse_patience", type=int, default=100)
    p.add_argument("--fine_epochs", type=int, default=2000)
    p.add_argument("--fine_patience", type=int, default=500)
    p.add_argument("--gpus", default="0", help="Comma list of CUDA ids, e.g. 0,1")
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--max_params", type=float, default=500_000,
                   help="Stop runs above this parameter count (0 = no limit)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retry_failed", action="store_true", help="Rerun cached failed runs")
    p.add_argument("--show_top", type=int, default=5, help="Configs printed per group")
    p.add_argument("--out_dir", default=None, help="Default: grid_<dataset>/")
    p.add_argument("--train_script", default="train_hop_masked_light.py")
    p.add_argument("--dry_run", action="store_true",
                   help="Print groups and run counts from the starting config, then exit")
    a = p.parse_args(argv)

    a.extra = extra
    a.fixed = {t[2:].split("=")[0] for t in extra if t.startswith("--")}
    # noemit: fixed after "--" (dropped from the config); skip: not searched
    # (noemit + --exclude; excluded ones keep their --base / default value).
    a.noemit = a.fixed & set(DEFAULTS)
    fx = parse_flags(" ".join(shlex.quote(t) for t in extra))
    a.fixed_heads = int(fx["num_heads"]) if "num_heads" in fx else None
    a.fixed_hidden = int(fx["hidden_dim"]) if "hidden_dim" in fx else None
    if {"hidden_dim", "num_heads"} & a.fixed:
        a.noemit |= {"head_dim", "num_heads"}
    if "one_linear" in a.fixed:
        a.noemit |= {"one_linear_path", "one_linear_crosshop"}
    a.skip = {x.strip() for x in a.exclude.split(",") if x.strip()} | a.noemit
    if "hidden_dim" in a.skip:
        a.skip.add("head_dim")
    if a.direction is None:
        a.direction = ("minimize" if a.dataset.lower().startswith(("zinc", "peptides-struct"))
                       else "maximize")
    a.out_dir = a.out_dir or f"grid_{a.dataset.replace('/', '_')}"
    os.makedirs(a.out_dir, exist_ok=True)

    inc = build_base(a)
    coarse, fine = build_spaces(a)

    if a.dry_run:
        print(f"direction {a.direction}; coarse {a.coarse_epochs} ep, fine {a.fine_epochs} ep")
        print("coarse groups (counts from the starting config; later counts depend on "
              "earlier choices):")
        for n, sp in coarse:
            print(f"  {n:<18} {len(group_grid(sp, inc, a, False)):>4}  {list(sp)}")
        print("fine groups:")
        for n, sp in fine:
            print(f"  {n:<18} {len(group_grid(sp, inc, a, True)):>4}  {list(sp)}")
        return

    slots = queue.Queue()
    for g in a.gpus.split(","):
        for _ in range(a.jobs_per_gpu):
            slots.put(f"cuda:{g.strip()}")
    pool = cf.ThreadPoolExecutor(max_workers=slots.qsize())
    best = None
    try:
        for rnd in range(a.rounds):
            for name, sp in coarse:
                cands = group_grid(sp, inc, a, fine=False)
                if not cands:
                    continue
                r = run_group(f"{name} (round {rnd + 1})", cands, "coarse", a, slots, pool)
                if r:
                    inc, best = r["cfg"], r
        for name, sp in fine:
            cands = group_grid(sp, inc, a, fine=True)
            if not cands:
                continue
            r = run_group(name, cands, "fine", a, slots, pool)
            if r:
                inc, best = r["cfg"], r
    except KeyboardInterrupt:
        print("\ninterrupted; finished runs are cached, rerun to resume", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        raise

    pool.shutdown()
    print("\n==== final ====")
    if best:
        print(f"val {best['best_val']:.4f}  test {best['test_at_best']:.4f}  "
              f"({best['stage']}, {best['params_M']}M params)")
    full = [sys.executable, a.train_script, "--dataset", a.dataset] + \
        to_cli(inc, a, a.fine_epochs, a.fine_patience)
    cmd = " ".join(shlex.quote(x) for x in full)
    print(cmd)
    json.dump({"cfg": inc, "result": best, "cmd": cmd},
              open(os.path.join(a.out_dir, "best.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

