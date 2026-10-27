# run_synthetic_suite.sh
#!/usr/bin/env bash
# Generate the synthetic path-task suite and run every model ablation on it.
#
# Usage:  bash run_synthetic_suite.sh            # generate + train
#         GEN=0 bash run_synthetic_suite.sh      # train only (datasets exist)
#         TRAIN=0 bash run_synthetic_suite.sh    # generate only
#         SUMMARY_ONLY=1 bash run_synthetic_suite.sh
#         SECTIONS="A D" TASKS_D="3 6" bash run_synthetic_suite.sh
# Finished (dataset, ablation) runs are skipped on rerun (log contains RESULT);
# datasets already built with the current generator version are not rebuilt.
#
# Factors are varied one at a time around a base config:
#   family=tree, n=200, k=10, edge_dim=8, noise=0.
# Sections and their tasks (TASKS_<section>, default: every task that applies):
#   A  k sweep, trees (k=2,5,10,20 at n=200; k=40 at n=500)   1 3 4 5 6
#   A2 long paths, peptide_like (k=10,20,40)                    1 3 4 5 6 7
#   B  direct edge, trees                                        2
#   C  several shortest paths: grid, er, superpixel_like         7
#   D  graph families / presets (k=5; 4 for molecule_like)       1 2 3 4 5 6 7
#   E  edge feature dim 1,8,32 (trees, k=10)                     2 3 4 5 6
#   F  train label noise 0,0.1,0.3 (trees, k=10)                 1 2 3 4 5 6
#   G  graph size n=50,200,500 (trees, k=5)                      1 2 3 4 5 6
#   H  average degree 2,4,8 (er, k=3)                            1 2 3 4 5 6 7
#   I  several pairs with pair IDs (trees, 4 pairs, k in 2..10)  1 2 3 4 5 6
#   J  several sources (trees, 4 sources; nearest k=5,
#      weighted k in 1..10)                                      1 3 4 5 6
#   K  mixed k per graph (trees, k in 2..20)                     1 3 4 5 6
# Task 7 runs with both --path_mode canonical and invariant, and is skipped
# on families without tied shortest paths (tree, path, reddit_like).
# Task 4 is the order-comparison task (datasets named t4b_*).

set -uo pipefail

# ---------------- settings (override via env) ----------------
OUT=${OUT:-synthetic}                 # dataset root
LOGS=${LOGS:-synthetic_logs_v2}
RESULTS=${RESULTS:-synthetic_results_v2.jsonl}
SEEDS=${SEEDS:-1}
NUM_TRAIN=${NUM_TRAIN:-2000}
NUM_VAL=${NUM_VAL:-500}
NUM_TEST=${NUM_TEST:-500}
GEN_SEED=${GEN_SEED:-0}
GEN=${GEN:-1}
TRAIN=${TRAIN:-1}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}
SECTIONS=${SECTIONS:-"A A2 B C D E F G H I J K"}

TASKS_A=${TASKS_A:-"1 3 4 5 6"}
TASKS_A2=${TASKS_A2:-"1 3 4 5 6 7"}
TASKS_B=${TASKS_B:-"2"}
TASKS_C=${TASKS_C:-"7"}
TASKS_D=${TASKS_D:-"1 2 3 4 5 6 7"}
TASKS_E=${TASKS_E:-"2 3 4 5 6"}
TASKS_F=${TASKS_F:-"1 2 3 4 5 6"}
TASKS_G=${TASKS_G:-"1 2 3 4 5 6"}
TASKS_H=${TASKS_H:-"1 2 3 4 5 6 7"}
TASKS_I=${TASKS_I:-"1 2 3 4 5 6"}
TASKS_J=${TASKS_J:-"1 3 4 5 6"}
TASKS_K=${TASKS_K:-"1 3 4 5 6"}
NUM_PAIRS=${NUM_PAIRS:-4}
NUM_SOURCES=${NUM_SOURCES:-4}

# Model / training (shared by all ablations). max_hops is set per dataset:
# clamp(ceil(diameter) + 1, max s-t distance + 1, MAX_HOPS).
MAX_HOPS=${MAX_HOPS:-48}
HIDDEN=${HIDDEN:-64}
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-4}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}
BATCH=${BATCH:-128}
# Smaller batches for large graphs (dense (B, K, N, N) masks and (B, N, N, d) P).
BATCH_LARGE=${BATCH_LARGE:-32}         # avg_nodes >= LARGE_N
LARGE_N=${LARGE_N:-400}
LR=${LR:-1e-3}
DROPOUT=${DROPOUT:-0.1}
WORKERS=${WORKERS:-4}
EXTRA=${EXTRA:-"--amp"}                      # extra flags for every run, e.g. "--amp"

COMMON="--seeds $SEEDS --auto_max_hops --max_hops_cap $MAX_HOPS --hidden_dim $HIDDEN --num_heads $HEADS \
--num_layers $LAYERS --max_epochs $EPOCHS --patience $PATIENCE \
--lr $LR --dropout $DROPOUT --num_workers $WORKERS --hop_mode contiguous \
--results $RESULTS $EXTRA"

# Ablations (name -> flags).
# --use_edge_features adds incident-edge means to node inputs (encoder);
# path_only isolates the path embeddings from that encoder step.
ABLATIONS=(node_only hop_mask hop_edge path_only full hop_edge_gat)
abl_flags() {
  case "$1" in
    node_only)    echo "--num_global_heads $HEADS" ;;                 # no structure, no edges
    hop_mask)     echo "" ;;                                          # k-hop masks only
    hop_edge)     echo "--use_edge_features" ;;                       # + encoder edge features
    path_only)    echo "--use_path_embeddings --diagnostics" ;;       # path embeddings only
    full)         echo "--use_edge_features --use_path_embeddings --diagnostics" ;;
    hop_edge_gat) echo "--use_edge_features --num_post_gat_layers 2" ;; # + MPNN layers
  esac
}

batch_for() {  # batch size from the dataset's mean node count
  python -c "import json,sys; n=json.load(open(sys.argv[1]))['stats']['avg_nodes']; \
print(sys.argv[3] if n >= float(sys.argv[2]) else sys.argv[4])" \
    "$1/meta.json" "$LARGE_N" "$BATCH_LARGE" "$BATCH"
}

NO_TIES="tree path reddit_like"            # unique shortest paths: task 7 infeasible

gen() {  # gen <task> <family> [extra generator args...]
  python synthetic_suite.py --task "$1" --family "$2" "${@:3}" --out "$OUT" \
    --num_train "$NUM_TRAIN" --num_val "$NUM_VAL" --num_test "$NUM_TEST" \
    --seed "$GEN_SEED" --skip_existing
}

gen_tasks() {  # gen_tasks "<tasks>" <family> [extra generator args...]
  local tasks=$1 fam=$2; shift 2
  for t in $tasks; do
    if [[ $t == 7 ]]; then
      [[ " $NO_TIES " == *" $fam "* ]] && continue
      gen 7 "$fam" "$@" --path_mode canonical
      gen 7 "$fam" "$@" --path_mode invariant
    else
      gen "$t" "$fam" "$@"
    fi
  done
}

has() { [[ " $SECTIONS " == *" $1 "* ]]; }

# ---------------- 1. generate ----------------
if [[ "$GEN" == 1 && "$SUMMARY_ONLY" == 0 ]]; then
  FAMS="tree path ring grid er ba peptide_like molecule_like tsp_like superpixel_like reddit_like dense_like community_like"

  # A. Path length k on trees; k=40 needs larger trees.
  if has A; then
    gen_tasks "$TASKS_A" tree --n 200 --k 2,5,10,20 --edge_dim 8
    gen_tasks "$TASKS_A" tree --n 500 --k 40 --edge_dim 8
  fi
  # A2. Long paths on peptide-like graphs (the target regime).
  has A2 && gen_tasks "$TASKS_A2" peptide_like --k 10,20,40 --edge_dim 8
  # B. Direct edge (k=1).
  has B && gen_tasks "$TASKS_B" tree --n 200 --edge_dim 8
  # C. Multiple shortest paths.
  if has C; then
    for fam in grid er superpixel_like; do
      gen_tasks "$TASKS_C" $fam --k 5,10 --edge_dim 8
    done
  fi
  # D. Graph families / presets (presets use their own n, deg).
  if has D; then
    for fam in $FAMS; do
      [[ $fam == molecule_like ]] && k=4 || k=5
      gen_tasks "$TASKS_D" $fam --k $k --edge_dim 8
    done
  fi
  # E. Edge feature dim.
  has E && gen_tasks "$TASKS_E" tree --n 200 --k 10 --edge_dim 1,8,32
  # F. Label noise (train only).
  has F && gen_tasks "$TASKS_F" tree --n 200 --k 10 --edge_dim 8 --noise 0,0.1,0.3
  # G. Graph size.
  has G && gen_tasks "$TASKS_G" tree --n 50,200,500 --k 5 --edge_dim 8
  # H. Average degree (ER, short k so dense graphs stay feasible).
  has H && gen_tasks "$TASKS_H" er --n 200 --deg 2,4,8 --k 3 --edge_dim 8
  # I. Several (s, t) pairs linked by pair IDs, a different k per pair.
  has I && gen_tasks "$TASKS_I" tree --n 200 --k_range 2,10 --pairs "$NUM_PAIRS" --edge_dim 8
  # J. Several sources, partner defined by structure (no IDs).
  if has J; then
    gen_tasks "$TASKS_J" tree --n 200 --k 5 --sources "$NUM_SOURCES" --rule nearest --edge_dim 8
    gen_tasks "$TASKS_J" tree --n 200 --k_range 1,10 --sources "$NUM_SOURCES" --rule weighted --edge_dim 8
  fi
  # K. Mixed k per graph.
  has K && gen_tasks "$TASKS_K" tree --n 200 --k_range 2,20 --edge_dim 8
fi

# ---------------- 2. train every ablation on every dataset ----------------
if [[ "$TRAIN" == 1 && "$SUMMARY_ONLY" == 0 ]]; then
  mkdir -p "$LOGS"
  for d in "$OUT"/*/; do
    d=${d%/}
    [[ -f "$d/meta.json" ]] || continue
    # Only datasets from the current generator version (older ones, e.g. the
    # ramp task 4 "t4_*", are stale).
    grep -q '"gen_version": 2' "$d/meta.json" || { echo "skip $(basename "$d") (old generator)"; continue; }
    name=$(basename "$d")
    bs=$(batch_for "$d")
    for abl in "${ABLATIONS[@]}"; do
      log="$LOGS/${name}__${abl}.log"
      if [[ -f "$log" ]] && grep -q "^RESULT" "$log"; then
        echo "skip $name / $abl (done)"; continue
      fi
      echo "=== $name / $abl (batch $bs) ==="
      # shellcheck disable=SC2086
      python run_synthetic.py --data "$d" $COMMON --batch_size "$bs" $(abl_flags "$abl") \
        > "$log" 2>&1 || echo "FAILED $name / $abl (see $log)"
      grep -E "^RESULT|path gates" "$log" || tail -n 3 "$log"
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
        return "full" if a.get("use_edge_features") else "path_only"
    if a.get("num_post_gat_layers", 0) > 0:
        return "hop_edge_gat"
    if a.get("use_edge_features"):
        return "hop_edge"
    if a.get("num_global_heads", 0) >= heads:
        return "node_only"
    return "hop_mask"

cols = ["node_only", "hop_mask", "hop_edge", "path_only", "full", "hop_edge_gat"]
table = defaultdict(dict)
gates = defaultdict(dict)
for line in open(path):
    r = json.loads(line)
    abl = ablation(r["args"])
    table[r["dataset"]][abl] = f"{r['test_mean']:.3f}±{r['test_std']:.3f}"  # last run wins
    g = [abs(v) for run in r.get("runs", []) for v in run.get("path_gates", [])]
    if g:
        gates[r["dataset"]][abl] = max(g)

w = max([len(k) for k in table] + [10])
print(f"\n{'dataset':<{w}}  " + "  ".join(f"{c:>13}" for c in cols))
for ds in sorted(table):
    print(f"{ds:<{w}}  " + "  ".join(f"{table[ds].get(c, '-'):>13}" for c in cols))
print("\nTest accuracy (mean±std over seeds); chance = 0.5. "
      "Tasks 2-7 should be ~0.5 for node_only, otherwise there is a label leak.")

if gates:
    print(f"\n{'dataset':<{w}}  {'max|gate| path_only':>20}  {'max|gate| full':>15}")
    for ds in sorted(gates):
        po, fu = gates[ds].get("path_only"), gates[ds].get("full")
        print(f"{ds:<{w}}  {('%.4f' % po) if po is not None else '-':>20}  "
              f"{('%.4f' % fu) if fu is not None else '-':>15}")
    print("Path gates start at 0; values near 0 mean the path term is unused.")
EOF

