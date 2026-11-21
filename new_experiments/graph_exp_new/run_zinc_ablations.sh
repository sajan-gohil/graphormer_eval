#!/usr/bin/env bash
# ZINC12k ablations around the best earlier configuration (hidden 64, 4 heads
# incl. 1 global, 4 layers, max_hops 10, path embeddings, linear head,
# --one_linear); each run changes one thing, plus one combined run. Runs are spread over the GPUs in $GPUS (one job per GPU at a time),
# finished runs (log has a BEST: line) are skipped on rerun, and a summary
# table is printed at the end.
#
# Usage:  bash run_zinc_ablations.sh
#         GPUS="0 1 2 3" SEEDS="0 1 2 3" bash run_zinc_ablations.sh
#         ONLY="base sched_plateau head_nonorm" bash run_zinc_ablations.sh
#         SUMMARY_ONLY=1 bash run_zinc_ablations.sh

set -uo pipefail

GPUS=${GPUS:-"0 1 2"}
SEEDS=${SEEDS:-"0"}
ONLY=${ONLY:-}                       # space-separated ablation names (empty = all)
LOGS=${LOGS:-zinc_ablation_logs_h64}
CKPT=${CKPT:-zinc_ablation_ckpt_h64}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}
WORKERS=${WORKERS:-4}

# Fixed settings (shared by every run): the best earlier ZINC run.
COMMON="--dataset ZINC12k --max_hops 10 --hidden_dim 64 --num_heads 4 --ffn_ratio 4 \
--num_layers 4 --dropout 0.2 --graph_pool sum --norm_type layer --hop_mode contiguous \
--hop_window 1 --num_global_heads 1 --mask_type shortest_path --lr 0.001 --lr_min 1e-6 \
--weight_decay 1e-5 --batch_size 512 --max_epochs 2000 --patience 500 \
--reduce_lr_patience 10 --warmup_ratio 0.01 --grad_clip 5.0 --random_global_prob 0.05 \
--num_workers $WORKERS"
P="--use_path_embeddings"
B="$P --linear_head --one_linear"      # base: path embeddings, linear head, 1-layer path term

# name | flags on top of COMMON (each changes one thing vs base unless noted).
# --one_linear = 1-layer path term AND 1-layer cross-hop output; the
# path/crosshop-specific variants drop it and set one of the two flags.
ABLATIONS=(
  "base|$B"
  "no_path|--linear_head --one_linear"
  "sched_plateau|$B --lr_scheduler plateau"
  "sched_constant|$B --lr_scheduler constant"
  "sched_cos1000|$B --cosine_epochs 1000"
  "head_mlp|$P --one_linear"
  "head_nodrop|$B --head_dropout 0"
  "head_nonorm|$B --head_norm none"
  "final_norm|$B --final_norm"
  "final_norm_head_nonorm|$B --final_norm --head_norm none"
  "pool_mean|$B --graph_pool mean"
  "path_norm_term|$B --path_norm term"
  "path_norm_none|$B --path_norm none"
  "path_two_layer|$P --linear_head"
  "path_edge_only|$B --path_node_terms edge"
  "path_edge_src|$B --path_node_terms edge,src"
  "path_edge_dst|$B --path_node_terms edge,dst"
  "path_nodes_only|$B --path_node_terms src,dst"
  "path_ties_uniform|$B --path_ties uniform"
  "crosshop|$B --dynamic_cross_hop"
  "crosshop_two_layer|$P --linear_head --one_linear_path --dynamic_cross_hop"
  # Combined: 2-layer cross-hop mixer + no path LayerNorm + 1-layer path term.
  "combined|$P --linear_head --one_linear_path --dynamic_cross_hop --path_norm none"
)

mkdir -p "$LOGS" "$CKPT"
read -r -a GPU_LIST <<< "$GPUS"
declare -A SLOT_PID=()

wait_for_slot() {   # prints a free GPU id
  while true; do
    for g in "${GPU_LIST[@]}"; do
      pid=${SLOT_PID[$g]:-}
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$g"; return
      fi
    done
    sleep 20
  done
}

wanted() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }

if [[ "$SUMMARY_ONLY" == 0 ]]; then
  for seed in $SEEDS; do
    for entry in "${ABLATIONS[@]}"; do
      name=${entry%%|*}; flags=${entry#*|}
      wanted "$name" || continue
      run="${name}_s${seed}"
      log="$LOGS/$run.log"
      if [[ -f "$log" ]] && grep -q "^BEST:" "$log"; then
        echo "skip $run (done)"; continue
      fi
      gpu=$(wait_for_slot)
      echo "start $run on cuda:$gpu"
      # shellcheck disable=SC2086
      OMP_NUM_THREADS=1 nohup python train_hop_masked_light.py $COMMON $flags \
        --seed "$seed" --device "cuda:$gpu" --save_dir "$CKPT/$run" \
        > "$log" 2>&1 &
      SLOT_PID[$gpu]=$!
      sleep 5
    done
  done
  wait
fi

# ---------------- summary ----------------
python - "$LOGS" <<'EOF'
import glob, os, re, sys
from collections import defaultdict
logs = sys.argv[1]
pat = re.compile(r"^BEST: val MAE ([0-9.]+) \| test MAE ([0-9.]+) \(epoch (\d+)\)")
par = re.compile(r"trainable params:\s*([0-9.]+)M")
res = defaultdict(list)
for f in sorted(glob.glob(os.path.join(logs, "*.log"))):
    name = os.path.basename(f)[:-4].rsplit("_s", 1)[0]
    txt = open(f).read().splitlines()
    best = [pat.match(l) for l in txt]
    best = [m for m in best if m]
    p = [par.search(l) for l in txt]
    p = next((m.group(1) for m in p if m), "-")
    if best:
        m = best[-1]
        res[name].append((float(m.group(1)), float(m.group(2)), int(m.group(3)), p))
    else:
        res[name].append(None)
rows = []
for name, runs in res.items():
    done = [r for r in runs if r]
    if not done:
        rows.append((name, None)); continue
    v = [r[0] for r in done]; t = [r[1] for r in done]
    mean = lambda x: sum(x) / len(x)
    sd = lambda x: (sum((a - mean(x)) ** 2 for a in x) / len(x)) ** 0.5
    rows.append((name, (mean(v), sd(v), mean(t), sd(t), len(done), len(runs),
                        int(mean([r[2] for r in done])), done[0][3])))
rows.sort(key=lambda r: (r[1] is None, r[1][0] if r[1] else 0))
print(f"\n{'ablation':<24} {'val MAE':>15} {'test MAE':>15} {'runs':>6} {'best ep':>8} {'params':>7}")
for name, r in rows:
    if r is None:
        print(f"{name:<24} {'(running/failed)':>15}"); continue
    print(f"{name:<24} {r[0]:>8.4f}±{r[1]:.4f} {r[2]:>8.4f}±{r[3]:.4f} {r[4]:>3}/{r[5]:<2} "
          f"{r[6]:>8} {r[7]:>6}M")
print("\nSorted by val MAE (lower is better). Std over seeds.")
EOF

