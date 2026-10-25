# run_synthetic_binding_tests.sh
#!/usr/bin/env bash
# Tests for the path-embedding change MLP([x_u ; e_uv ; x_v ; pos_r]) (edge/node
# content bound to its position before the path sum).
#
#   O  order (task 4b): should rise well above chance; tree k=2 should approach 1.0
#   Q  query lookup (task 5): expected to stay near chance (no query conditioning
#      yet); any rise above the averaging ceiling 0.5+asin(1/sqrt(k))/pi is a bonus
#   R  regression (tasks 1, 2, 6): must not drop vs the previous results
#   H  hard cases from the first suite, rerun with more training and 3 seeds
#      (separates training failures from model limits)
#
# Datasets are reused from $OUT if already generated (same names as the suite).
# Results go to a new log/results file; the summary compares path_only against
# the previous results file ($PREV_RESULTS) where the dataset exists in both.
#
# Usage:  bash run_synthetic_binding_tests.sh
#         SECTIONS="O R" SEEDS=0 bash run_synthetic_binding_tests.sh
#         SUMMARY_ONLY=1 bash run_synthetic_binding_tests.sh

set -uo pipefail

# ---------------- settings (override via env) ----------------
OUT=${OUT:-synthetic}
LOGS=${LOGS:-synthetic_logs_binding}
RESULTS=${RESULTS:-synthetic_results_binding.jsonl}
PREV_RESULTS=${PREV_RESULTS:-synthetic_results_v2.jsonl}
DSLIST=${DSLIST:-$LOGS/datasets.txt}
SECTIONS=${SECTIONS:-"O Q R H"}
ABLATIONS=(${ABLATIONS:-hop_mask path_only full hop_edge_gat})
SEEDS=${SEEDS:-0,1,2}
NUM_TRAIN=${NUM_TRAIN:-2000}
NUM_VAL=${NUM_VAL:-500}
NUM_TEST=${NUM_TEST:-500}
GEN_SEED=${GEN_SEED:-0}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}

# Longer training than the first suite (runs there often stopped at a shortcut).
MAX_HOPS=${MAX_HOPS:-48}
HIDDEN=${HIDDEN:-64}
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-4}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-40}
BATCH=${BATCH:-32}
BATCH_LARGE=${BATCH_LARGE:-8}
LARGE_N=${LARGE_N:-400}
LR=${LR:-1e-3}
DROPOUT=${DROPOUT:-0.1}
WORKERS=${WORKERS:-4}
EXTRA=${EXTRA:-}

COMMON="--seeds $SEEDS --auto_max_hops --max_hops_cap $MAX_HOPS --hidden_dim $HIDDEN \
--num_heads $HEADS --num_layers $LAYERS --max_epochs $EPOCHS --patience $PATIENCE \
--lr $LR --dropout $DROPOUT --num_workers $WORKERS --hop_mode contiguous \
--results $RESULTS $EXTRA"

abl_flags() {
  case "$1" in
    node_only)    echo "--num_global_heads $HEADS" ;;
    hop_mask)     echo "" ;;
    hop_edge)     echo "--use_edge_features" ;;
    path_only)    echo "--use_path_embeddings --diagnostics" ;;
    full)         echo "--use_edge_features --use_path_embeddings --diagnostics" ;;
    hop_edge_gat) echo "--use_edge_features --num_post_gat_layers 2" ;;
  esac
}

batch_for() {
  python -c "import json,sys; n=json.load(open(sys.argv[1]))['stats']['avg_nodes']; \
print(sys.argv[3] if n >= float(sys.argv[2]) else sys.argv[4])" \
    "$1/meta.json" "$LARGE_N" "$BATCH_LARGE" "$BATCH"
}

has() { [[ " $SECTIONS " == *" $1 "* ]]; }

# gen <task> <family> [args...]: generate (or reuse) and record dataset dirs.
gen() {
  python synthetic_suite.py --task "$1" --family "$2" "${@:3}" --out "$OUT" \
    --num_train "$NUM_TRAIN" --num_val "$NUM_VAL" --num_test "$NUM_TEST" \
    --seed "$GEN_SEED" --skip_existing 2>&1 | tee -a "$LOGS/generate.log" | \
  python -c "
import sys, os
out = sys.argv[1]
for line in sys.stdin:
    line = line.rstrip()
    if line.startswith('[synthetic] exists '):
        print(os.path.join(out, line.split(' ', 2)[2]))
    elif line.startswith('[synthetic] ') and ': {' in line:
        print(line[len('[synthetic] '):].split(': {')[0])
    elif line.startswith('[synthetic] skipped'):
        print(line, file=sys.stderr)
" "$OUT" >> "$DSLIST"
}

# ---------------- 1. generate / collect datasets ----------------
mkdir -p "$LOGS"
if [[ "$SUMMARY_ONLY" == 0 ]]; then
  : > "$DSLIST"

  if has O; then   # order: k sweep on trees, long paths, several families
    gen 4 tree --n 200 --k 2,5,10,20 --edge_dim 8
    gen 4 tree --n 500 --k 40 --edge_dim 8
    gen 4 peptide_like --k 5,10,20,40 --edge_dim 8
    for fam in molecule_like reddit_like path grid superpixel_like er; do
      [[ $fam == molecule_like ]] && k=4 || k=5
      [[ $fam == er ]] && extra="--deg 2" k=3 || extra=""
      # shellcheck disable=SC2086
      gen 4 $fam --k $k --edge_dim 8 $extra
    done
    gen 4 tree --n 200 --k 10 --edge_dim 1,32
  fi

  if has Q; then   # query lookup
    gen 5 tree --n 200 --k 2,5,10 --edge_dim 8
    gen 5 peptide_like --k 10 --edge_dim 8
  fi

  if has R; then   # regression on tasks that worked before
    gen 1 tree --n 200 --k 10 --edge_dim 8
    gen 1 tree --n 200 --k 5 --sources 4 --rule nearest --edge_dim 8
    gen 2 tree --n 200 --edge_dim 8
    gen 2 superpixel_like --edge_dim 8
    gen 6 tree --n 200 --k 5,10 --edge_dim 8
    gen 6 peptide_like --k 10,40 --edge_dim 8
    gen 6 superpixel_like --k 5 --edge_dim 8
    gen 6 tsp_like --k 5 --edge_dim 8
    gen 6 er --n 200 --deg 2 --k 3 --edge_dim 8
  fi

  if has H; then   # hard cases (training vs model limit), more training + 3 seeds
    gen 1 er --n 200 --deg 8 --k 3 --edge_dim 8
    gen 6 er --n 200 --deg 4,8 --k 3 --edge_dim 8
    gen 1 tree --n 200 --k_range 2,20 --edge_dim 8
    gen 6 tree --n 200 --k_range 2,20 --edge_dim 8
    gen 1 tree --n 200 --k 10 --edge_dim 8 --noise 0.1
    gen 6 tree --n 200 --k 10 --edge_dim 8 --noise 0.1
  fi

  sort -u -o "$DSLIST" "$DSLIST"
  echo "$(wc -l < "$DSLIST") datasets listed in $DSLIST"

  # ---------------- 2. train ----------------
  while read -r d; do
    [[ -f "$d/meta.json" ]] || { echo "missing $d"; continue; }
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
  done < "$DSLIST"
fi

# ---------------- 3. summary ----------------
[[ -f "$RESULTS" ]] || { echo "no results file $RESULTS"; exit 0; }
python - "$RESULTS" "$PREV_RESULTS" "$HEADS" <<'EOF'
import json, math, os, re, sys
from collections import defaultdict

path, prev_path, heads = sys.argv[1], sys.argv[2], int(sys.argv[3])

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

def load(p):
    tab = defaultdict(dict)
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            tab[r["dataset"]][ablation(r["args"])] = r       # last run wins
    return tab

cur, prev = load(path), load(prev_path)
cols = ["node_only", "hop_mask", "hop_edge", "path_only", "full", "hop_edge_gat"]
cols = [c for c in cols if any(c in v for v in cur.values())]

def cell(r):
    return f"{r['test_mean']:.3f}±{r['test_std']:.3f}" if r else "-"

w = max([len(k) for k in cur] + [10])
print(f"\n{'dataset':<{w}}  " + "  ".join(f"{c:>13}" for c in cols)
      + f"  {'path_only prev':>14}  {'Δ':>7}")
for ds in sorted(cur):
    po, pp = cur[ds].get("path_only"), prev.get(ds, {}).get("path_only")
    delta = f"{po['test_mean'] - pp['test_mean']:+.3f}" if po and pp else "-"
    prevs = f"{pp['test_mean']:.3f}" if pp else "-"
    print(f"{ds:<{w}}  " + "  ".join(f"{cell(cur[ds].get(c)):>13}" for c in cols)
          + f"  {prevs:>14}  {delta:>7}")

# Expectations for the binding change (path_only).
print("\nChecks (path_only):")
def k_of(ds):
    m = re.search(r"_k(\d+)_", ds)
    return int(m.group(1)) if m else None
for ds in sorted(cur):
    po = cur[ds].get("path_only")
    if not po:
        continue
    acc, k = po["test_mean"], k_of(ds)
    if ds.startswith("t4b_"):
        status = "PASS" if acc >= 0.75 else ("partial" if acc >= 0.6 else "FAIL")
        print(f"  order   {ds}: {acc:.3f}  [{status}]  (chance 0.5; target >= 0.75, k=2 ~1.0)")
    elif ds.startswith("t5_") and k:
        ceil = 0.5 + math.asin(1 / math.sqrt(k)) / math.pi
        status = "above ceiling" if acc > ceil + 0.03 else "at/below averaging ceiling"
        print(f"  query   {ds}: {acc:.3f} vs averaging ceiling {ceil:.3f}  [{status}]")
    else:
        pp = prev.get(ds, {}).get("path_only")
        if pp:
            d = acc - pp["test_mean"]
            status = "ok" if d > -0.03 else "REGRESSION"
            print(f"  regress {ds}: {acc:.3f} vs prev {pp['test_mean']:.3f} ({d:+.3f})  [{status}]")

gates = {ds: max(abs(v) for run in r["path_only"].get("runs", [])
                 for v in run.get("path_gates", [0]))
         for ds, r in cur.items() if "path_only" in r}
if gates:
    print("\nmax|gate| path_only (0 = path term unused):")
    for ds in sorted(gates):
        print(f"  {ds:<{w}}  {gates[ds]:.4f}")
EOF

