#!/bin/bash
# =============================================================================
# RETRY: Failed Ablation Experiments - num_heads Fixed
# =============================================================================
#
# Fix applied: for single hop_mode,
#   num_heads = num_global_heads + (max_hops - 1)
#
#   max_hops=30, global=0  -> num_heads = 0 + 29 = 29
#   max_hops=30, global=1  -> num_heads = 1 + 29 = 30  (baseline)
#   max_hops=30, global=2  -> num_heads = 2 + 29 = 31
#   max_hops=30, global=3  -> num_heads = 3 + 29 = 32
#   max_hops=30, global=5  -> num_heads = 5 + 29 = 34
#
# For G16a/G16b (smaller width), max_hops is also reduced to satisfy:
#   max_hops = num_heads - num_global_heads + 1
#   G16a: 16 - 1 + 1 = 16  -> --max_hops 16
#   G16b: 20 - 1 + 1 = 20  -> --max_hops 20
#
# Runs 6 experiments in parallel: 3 on cuda:1 + 3 on cuda:2.
# =============================================================================

SCRIPT="train_hop_masked_transformer_final.py"
DATASET="Peptides-func"
LOG_DIR="ablation_logs_func_patient"
CKPT_DIR="ablation_checkpoints_patient"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

GPUS=("cuda:2")
JOBS_PER_GPU=1

BASE_MAX_HOPS=30
BASE_NUM_GLOBAL_HEADS=1
BASE_HOP_MODE="single"
BASE_HOP_WINDOW=0
BASE_HIDDEN_DIM=240
BASE_NUM_LAYERS=1
BASE_FFN_RATIO=1
BASE_DROPOUT=0.2
BASE_LR=1e-3
BASE_BATCH_SIZE=200
BASE_MAX_EPOCHS=1000
BASE_PATIENCE=200
BASE_GRAD_CLIP=5.0
BASE_SEED=0

COMMON="--dataset ${DATASET} \
  --max_hops ${BASE_MAX_HOPS} \
  --num_layers ${BASE_NUM_LAYERS} \
  --ffn_ratio ${BASE_FFN_RATIO} \
  --dropout ${BASE_DROPOUT} \
  --lr ${BASE_LR} \
  --batch_size ${BASE_BATCH_SIZE} \
  --max_epochs ${BASE_MAX_EPOCHS} \
  --patience ${BASE_PATIENCE} \
  --grad_clip ${BASE_GRAD_CLIP} \
  --seed ${BASE_SEED} \
  --num_workers 0"

QUEUE=()

enqueue() {
    local name="$1"
    shift
    QUEUE+=("${name}|$*")
}

flush_queue() {
    local batch_size=$(( JOBS_PER_GPU * ${#GPUS[@]} ))
    local total=${#QUEUE[@]}
    local batch_num=0

    for (( start=0; start<total; start+=batch_size )); do
        batch_num=$((batch_num + 1))
        local end=$(( start + batch_size ))
        if (( end > total )); then end=$total; fi
        local count=$(( end - start ))

        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  BATCH ${batch_num}: launching ${count} experiments in parallel"
        echo "║  (experiments $((start+1))-${end} of ${total})"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo ""

        local pids=()
        local names=()
        local gpu_idx=0
        local gpu_slot=0

        for (( i=start; i<end; i++ )); do
            local entry="${QUEUE[$i]}"
            local name="${entry%%|*}"
            local extra="${entry#*|}"
            local device="${GPUS[$gpu_idx]}"
            local logfile="${LOG_DIR}/${name}.log"

            if [ -f "${logfile}" ] && grep -q '^BEST:' "${logfile}" 2>/dev/null; then
                echo "  SKIP (already done): ${name}"
                continue
            fi

            echo "  START: ${name}  [${device}]  -> ${logfile}"

            mkdir -p "${CKPT_DIR}/${name}"
            python3 ${SCRIPT} ${COMMON} \
                --device "${device}" \
                --save_dir "${CKPT_DIR}/${name}" \
                ${extra} \
                > "${logfile}" 2>&1 &

            pids+=($!)
            names+=("${name}")

            gpu_slot=$((gpu_slot + 1))
            if (( gpu_slot >= JOBS_PER_GPU )); then
                gpu_slot=0
                gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
            fi
        done

        if [ ${#pids[@]} -gt 0 ]; then
            echo ""
            echo "  Waiting for ${#pids[@]} jobs..."
            local failed=0
            for idx in "${!pids[@]}"; do
                wait "${pids[$idx]}"
                local rc=$?
                local n="${names[$idx]}"
                local lf="${LOG_DIR}/${n}.log"
                if [ $rc -ne 0 ]; then
                    echo "  x FAILED (rc=${rc}): ${n}"
                    failed=$((failed + 1))
                else
                    local best=$(grep '^BEST:' "${lf}" 2>/dev/null || echo "NO BEST LINE")
                    echo "  + DONE: ${n}  >>  ${best}"
                fi
            done
            echo "  Batch ${batch_num} complete (${#pids[@]} ran, ${failed} failed)."
        fi
    done

    QUEUE=()
}

# =============================================================================
# BATCH 1 (6 experiments):
#   G1b_hop_mode_window_w1  (num_heads=30 = 1+29)
#   G2a_global_heads_0      (num_heads=29 = 0+29)
#   G2b_global_heads_2      (num_heads=31 = 2+29)
#   G2c_global_heads_3      (num_heads=32 = 3+29)
#   G2d_global_heads_5      (num_heads=34 = 5+29)
#   G3a_moe_dense           (num_heads=30 = 1+29)
# =============================================================================
echo "===== GROUP 1 (retry): HOP MODE WINDOW W1 ====="
echo "===== GROUP 1 (retry): HOP MODE WINDOW W1 ====="

#enqueue "C1_sparse_structural_shallow" \
#    --hop_mode ${BASE_HOP_MODE} \
#    --hop_window ${BASE_HOP_WINDOW} \
#    --num_layers 2 \
#    --hidden_dim 160 \
#    --num_heads 20 \
#    --max_hops 20 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --block_diag_out \
#    --blend_adj_power

#enqueue "C2_moe_dynamic_shallow" \
#    --hop_mode ${BASE_HOP_MODE} \
#    --hop_window ${BASE_HOP_WINDOW} \
#    --num_layers 2 \
#    --hidden_dim 240 \
#    --num_heads 30 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --use_moe_gating \
#    --top_k 1 \
#    --block_diag_out

#enqueue "C3_cross_hop_laplacian" \
#    --hop_mode ${BASE_HOP_MODE} \
#    --hop_window ${BASE_HOP_WINDOW} \
#    --num_layers 2 \
#    --hidden_dim 240 \
#    --num_heads 30 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --dynamic_cross_hop \
#    --block_diag_out \
#    --cross_hop_no_ffn \
#    --use_edge_features \
#    --use_lap_pe \
#    --lap_pe_dim 8


#enqueue "C4_sparse_structural_shallow" \
#    --hop_mode ${BASE_HOP_MODE} \
#    --hop_window ${BASE_HOP_WINDOW} \
#    --num_layers 3 \
#    --ffn_ratio 1 \
#    --hidden_dim 160 \
#    --num_heads 20 \
#    --max_hops 20 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --block_diag_out \
#    --blend_adj_power

enqueue "C5_moe_dynamic_shallow" \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_layers 3 \
    --ffn_ratio 1 \
    --hidden_dim 240 \
    --num_heads 30 \
    --num_global_heads 1 \
    --norm_type graph \
    --use_moe_gating \
    --top_k 1 \
    --block_diag_out \
    --batch_size 64

#enqueue "C6_cross_hop_laplacian" \
#    --hop_mode ${BASE_HOP_MODE} \
#    --hop_window ${BASE_HOP_WINDOW} \
#    --num_layers 3 \
#    --ffn_ratio 1 \
#    --hidden_dim 240 \
#    --num_heads 30 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --dynamic_cross_hop \
#    --block_diag_out \
#    --cross_hop_no_ffn \
#    --use_edge_features \
#    --use_lap_pe \
#    --lap_pe_dim 8


# 7. C1 + 40 heads (scaled hidden_dim and max_hops to preserve head capacity)
#enqueue "C1_heads_40_d160" \
#    --num_layers 2 \
#    --hidden_dim 160 \
#    --num_heads 40 \
#    --max_hops 40 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --block_diag_out \
#    --blend_adj_power

#enqueue "C_FINAL_ULTIMATE" \
#    --num_layers 2 \
#    --hidden_dim 160 \
#    --num_heads 40 \
#    --max_hops 40 \
#    --ffn_ratio 1 \
#    --num_global_heads 1 \
#    --norm_type graph \
#    --block_diag_out \
#    --use_pos_weight
    
flush_queue

# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              RETRY EXPERIMENTS COMPLETE                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results summary (experiment -> best test AP):"
echo "--------------------------------------------------------------"
for name in \
   C4_sparse_structural_shallow \
    C5_moe_dynamic_shallow \
    C1_heads_40_d160 \
    C_FINAL_ULTIMATE \
    C6_cross_hop_laplacian; do
    logfile="${LOG_DIR}/${name}.log"
    best=$(grep '^BEST:' "${logfile}" 2>/dev/null || echo "NO RESULT")
    printf "  %-45s %s\n" "${name}" "${best}"
done
echo "--------------------------------------------------------------"
echo ""
echo "Full logs in: ${LOG_DIR}/"
echo "Checkpoints in: ${CKPT_DIR}/"
 #   C1_sparse_structural_shallow \
 #   C2_moe_dynamic_shallow \
 #   C3_cross_hop_laplacian \
 
