# run_synthetic_suite.sh
#!/usr/bin/env bash
# Generate the synthetic path-task suite and run every model ablation on it.
#
# Usage:  bash run_synthetic_suite.sh            # generate + train
#         GEN=0 bash run_synthetic_suite.sh      # train only (datasets exist)
#         TRAIN=0 bash run_synthetic_suite.sh    # generate only
#         SUMMARY_ONLY=1 bash run_synthetic_suite.sh
# Finished (dataset, ablation) runs are skipped on rerun (log contains RESULT).
#
# Factors are varied one at a time around a base config:
#   family=tree, n=200, k=10, edge_dim=8, noise=0.

set -uo pipefail

# ---------------- settings (override via env) ----------------
OUT=${OUT:-synthetic}                 # dataset root
LOGS=${LOGS:-synthetic_logs}
RESULTS=${RESULTS:-synthetic_results.jsonl}
SEEDS=${SEEDS:-0,1,2}
NUM_TRAIN=${NUM_TRAIN:-2000}
NUM_VAL=${NUM_VAL:-500}
NUM_TEST=${NUM_TEST:-500}
GEN_SEED=${GEN_SEED:-0}
GEN=${GEN:-1}
TRAIN=${TRAIN:-1}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}

# Model / training (shared by all ablations). MAX_HOPS must exceed the largest k.
MAX_HOPS=${MAX_HOPS:-48}
HIDDEN=${HIDDEN:-64}
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-4}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}
BATCH=${BATCH:-32}
LR=${LR:-1e-3}
DROPOUT=${DROPOUT:-0.1}
WORKERS=${WORKERS:-4}
EXTRA=${EXTRA:-}                      # extra flags for every run, e.g. "--amp"

COMMON="--seeds $SEEDS --max_hops $MAX_HOPS --hidden_dim $HIDDEN --num_heads $HEADS \
--num_layers $LAYERS --max_epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH \
--lr $LR --dropout $DROPOUT --num_workers $WORKERS --hop_mode contiguous \
--results $RESULTS $EXTRA"

# Ablations (name -> flags).
ABLATIONS=(node_only hop_mask hop_edge full hop_edge_gat)
abl_flags() {
  case "$1" in
    node_only)    echo "--num_global_heads $HEADS" ;;                 # no structure, no edges
    hop_mask)     echo "" ;;                                          # k-hop masks only
    hop_edge)     echo "--use_edge_features" ;;                       # + edge features
    full)         echo "--use_edge_features --use_path_embeddings" ;; # + path embeddings
    hop_edge_gat) echo "--use_edge_features --num_post_gat_layers 2" ;; # + MPNN layers
  esac
}

gen() {  # gen <task> <family> [extra generator args...]
  python synthetic_suite.py --task "$1" --family "$2" "${@:3}" --out "$OUT" \
    --num_train "$NUM_TRAIN" --num_val "$NUM_VAL" --num_test "$NUM_TEST" --seed "$GEN_SEED"
}

# ---------------- 1. generate ----------------
if [[ "$GEN" == 1 && "$SUMMARY_ONLY" == 0 ]]; then
  FAMS="tree path ring grid er ba peptide_like molecule_like tsp_like superpixel_like reddit_like dense_like community_like"

  # A. Path length k (tasks 1, 3-6) on trees; k=40 needs larger trees.
  for t in 1 3 4 5 6; do
    gen $t tree --n 200 --k 2,5,10,20 --edge_dim 8
    gen $t tree --n 500 --k 40 --edge_dim 8
  done
  # A2. Long paths on peptide-like graphs (the target regime).
  for t in 3 4 5 6; do
    gen $t peptide_like --k 10,20,40 --edge_dim 8
  done
  # B. Direct edge (task 2, k=1).
  gen 2 tree --n 200 --edge_dim 8
  # C. Multiple shortest paths (task 7): canonical vs invariant labels.
  for fam in grid er superpixel_like; do
    for mode in canonical invariant; do
      gen 7 $fam --k 5,10 --edge_dim 8 --path_mode $mode
    done
  done
  # D. Graph families / presets (tasks 3 and 6, k=5; presets use their own n, deg).
  for t in 3 6; do
    for fam in $FAMS; do
      [[ $fam == molecule_like ]] && k=4 || k=5
      gen $t $fam --k $k --edge_dim 8
    done
  done
  # E. Edge feature dim.
  gen 3 tree --n 200 --k 10 --edge_dim 1,8,32
  # F. Label noise (train only).
  gen 3 tree --n 200 --k 10 --edge_dim 8 --noise 0,0.1,0.3
  # G. Graph size.
  gen 3 tree --n 50,200,500 --k 5 --edge_dim 8
  # H. Average degree (ER, short k so dense graphs stay feasible).
  gen 6 er --n 200 --deg 2,4,8 --k 3 --edge_dim 8
fi

# ---------------- 2. train every ablation on every dataset ----------------
if [[ "$TRAIN" == 1 && "$SUMMARY_ONLY" == 0 ]]; then
  mkdir -p "$LOGS"
  for d in "$OUT"/*/; do
    d=${d%/}
    [[ -f "$d/meta.json" ]] || continue
    name=$(basename "$d")
    for abl in "${ABLATIONS[@]}"; do
      log="$LOGS/${name}__${abl}.log"
      if [[ -f "$log" ]] && grep -q "^RESULT" "$log"; then
        echo "skip $name / $abl (done)"; continue
      fi
      echo "=== $name / $abl ==="
      # shellcheck disable=SC2086
      python run_synthetic.py --data "$d" $COMMON $(abl_flags "$abl") \
        > "$log" 2>&1 || echo "FAILED $name / $abl (see $log)"
      grep "^RESULT" "$log" || true
    done
  done
fi

# ---------------- 3. summary table ----------------
[[ -f "$RESULTS" ]] || { echo "no results file $RESULTS"; exit 0; }
python - "$RESULTS" "$HEADS" <<'EOF'
import json, sys
from collections import defaultdict

path, heads = sys.argv[1], int(sys.argv[2])

def ablation(a):
    if a.get("use_path_embeddings"):
        return "full"
    if a.get("num_post_gat_layers", 0) > 0:
        return "hop_edge_gat"
    if a.get("use_edge_features"):
        return "hop_edge"
    if a.get("num_global_heads", 0) >= heads:
        return "node_only"
    return "hop_mask"

cols = ["node_only", "hop_mask", "hop_edge", "full", "hop_edge_gat"]
table = defaultdict(dict)
for line in open(path):
    r = json.loads(line)
    table[r["dataset"]][ablation(r["args"])] = (
        f"{r['test_mean']:.3f}±{r['test_std']:.3f}")       # last run wins

w = max([len(k) for k in table] + [10])
print(f"\n{'dataset':<{w}}  " + "  ".join(f"{c:>13}" for c in cols))
for ds in sorted(table):
    print(f"{ds:<{w}}  " + "  ".join(f"{table[ds].get(c, '-'):>13}" for c in cols))
print("\nTest accuracy (mean±std over seeds); chance = 0.5. "
      "Tasks 3-7 should be ~0.5 for node_only, otherwise there is a label leak.")
EOF

