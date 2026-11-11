# run_zinc_ablations.sh
#!/usr/bin/env bash
# ZINC12k ablations around one fixed base configuration; each run changes one
# thing. Runs are spread over the GPUs in $GPUS (one job per GPU at a time),
# finished runs (log has a BEST: line) are skipped on rerun, and a summary
# table is printed at the end.
#
# Usage:  bash run_zinc_ablations.sh
#         GPUS="0 1 2 3" SEEDS="0 1 2 3" bash run_zinc_ablations.sh
#         ONLY="base sched_plateau head_nonorm" bash run_zinc_ablations.sh
#         SUMMARY_ONLY=1 bash run_zinc_ablations.sh

set -uo pipefail

GPUS=${GPUS:-"0"}
SEEDS=${SEEDS:-"0"}
ONLY=${ONLY:-}                       # space-separated ablation names (empty = all)
LOGS=${LOGS:-zinc_ablation_logs}
CKPT=${CKPT:-zinc_ablation_ckpt}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}
BATCH=${BATCH:-200}                  # not fixed by the base spec; best earlier run used 200
RGP=${RGP:-0.05}                     # random_global_prob; default would be dropout/2 = 0.1
WORKERS=${WORKERS:-4}

# Fixed base configuration.
BASE="--dataset ZINC12k --max_hops 8 --num_heads 8 --ffn_ratio 4 --num_global_heads 1 \
--num_layers 4 --hop_mode single --lr 0.001 --weight_decay 1e-5 --grad_clip 5 \
--max_epochs 2000 --patience 300 --dropout 0.2 --hidden_dim 128 \
--graph_pool sum --batch_size $BATCH --random_global_prob $RGP --num_workers $WORKERS"
P="--use_path_embeddings"            # on for every ablation except no_path

# name | extra flags (one change vs base, base = path embeddings + defaults:
# cosine over max_epochs, 2-layer MLP head with LayerNorm, path_norm sum,
# 2-layer path term, no cross-hop mixer)
ABLATIONS=(
  "base|$P"
  "no_path|"
  "sched_plateau|$P --lr_scheduler plateau"
  "sched_constant|$P --lr_scheduler constant"
  "sched_cos1000|$P --cosine_epochs 1000"
  "sched_cos600|$P --cosine_epochs 600"
  "head_linear|$P --linear_head"
  "head_linear_nodrop|$P --linear_head --head_dropout 0"
  "head_nodrop|$P --head_dropout 0"
  "head_nonorm|$P --head_norm none"
  "final_norm|$P --final_norm"
  "final_norm_head_nonorm|$P --final_norm --head_norm none"
  "pool_mean|$P --graph_pool mean"
  "path_norm_term|$P --path_norm term"
  "path_norm_none|$P --path_norm none"
  "one_linear_path|$P --one_linear_path"
  "path_edge_only|$P --path_node_terms edge"
  "crosshop|$P --dynamic_cross_hop"
  "crosshop_one_linear|$P --dynamic_cross_hop --one_linear_crosshop"
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
      OMP_NUM_THREADS=1 nohup python train_hop_masked_light.py $BASE $flags \
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

