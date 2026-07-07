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
LOG_DIR="ablation_logs"
CKPT_DIR="ablation_checkpoints"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

GPUS=("cuda:1" "cuda:2")
JOBS_PER_GPU=1

BASE_MAX_HOPS=30
BASE_NUM_GLOBAL_HEADS=1
BASE_HOP_MODE="single"
BASE_HOP_WINDOW=0
BASE_HIDDEN_DIM=240
BASE_NUM_LAYERS=4
BASE_FFN_RATIO=4
BASE_DROPOUT=0.2
BASE_LR=1e-3
BASE_BATCH_SIZE=64
BASE_MAX_EPOCHS=200
BASE_PATIENCE=40
BASE_GRAD_CLIP=1.0
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

enqueue "G1b_hop_mode_window_w1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode window \
    --hop_window 1

echo "===== GROUP 2 (retry): GLOBAL HEADS ABLATION ====="

enqueue "G2a_global_heads_0" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 29 \
    --num_global_heads 0 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

enqueue "G2b_global_heads_2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 31 \
    --num_global_heads 2 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

enqueue "G2c_global_heads_3" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 32 \
    --num_global_heads 3 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

enqueue "G2d_global_heads_5" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 34 \
    --num_global_heads 5 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# echo "===== GROUP 3 (retry): MoE GATING ====="

enqueue "G3a_moe_dense" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 0

flush_queue

# =============================================================================
# BATCH 2 (5 experiments):
#   G3c_moe_top3
#   G3d_moe_top5
#   G4a_multihop_sum
#   G4b_multihop_mean
#   G4c_multihop_sum_no_global
# =============================================================================
enqueue "G3c_moe_top3" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 3

enqueue "G3d_moe_top5" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 5

echo "===== GROUP 4 (retry): MULTIHOP ATTENTION ====="

enqueue "G4a_multihop_sum" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout sum

enqueue "G4b_multihop_mean" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout mean

enqueue "G4c_multihop_sum_no_global" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout sum \
    --multihop_no_global

flush_queue

# =============================================================================
# BATCH 3 (5 experiments):
#   G4d_multihop_mean_no_global
#   G5a_cross_hop
#   G5c_cross_hop_embed
#   G5e_cross_hop_full
#   G7a_mask_adj_power
# =============================================================================
enqueue "G4d_multihop_mean_no_global" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout mean \
    --multihop_no_global

echo "===== GROUP 5 (retry): CROSS-HOP MIXER ====="

enqueue "G5a_cross_hop" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop

enqueue "G5c_cross_hop_embed" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --cross_hop_hop_embedding

enqueue "G5e_cross_hop_full" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --block_diag_out \
    --cross_hop_hop_embedding \
    --cross_hop_no_ffn

# echo "===== GROUP 7 (retry): MASK TYPE ADJ_POWER ====="

enqueue "G7a_mask_adj_power" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --mask_type adj_power

# flush_queue

# # =============================================================================
# # BATCH 4 (5 experiments):
# #   G9c_gat_1layer_edge
# #   G14b_focal_g1
# #   G16a_h16_d128    (max_hops=16 to match 16-1+1=16)
# #   G16b_h20_d160    (max_hops=20 to match 20-1+1=20)
# #   G18e_kitchen_sink
# # =============================================================================
# echo "===== GROUP 9 (retry): GAT 1 LAYER + EDGE ====="

enqueue "G9c_gat_1layer_edge" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 4 \
    --use_edge_features

# echo "===== GROUP 14 (retry): FOCAL LOSS GAMMA=1 ====="

enqueue "G14b_focal_g1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --focal_gamma 1.0

# echo "===== GROUP 16 (retry): WIDTH ABLATION ====="

# # G16a: 16 heads, 1 global -> max_hops = 16 - 1 + 1 = 16
enqueue "G16a_h16_d128" \
    --hidden_dim 128 \
    --num_heads 16 \
    --num_global_heads 1 \
    --max_hops 16 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# # G16b: 20 heads, 1 global -> max_hops = 20 - 1 + 1 = 20
enqueue "G16b_h20_d160" \
    --hidden_dim 160 \
    --num_heads 20 \
    --num_global_heads 1 \
    --max_hops 20 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# echo "===== GROUP 18 (retry): KITCHEN SINK ====="

enqueue "G18e_kitchen_sink" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_lap_pe --lap_pe_dim 8 \
    --use_rrwp --rrwp_dim 8 \
    --blend_adj_power \
    --use_virtual_node

flush_queue

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              RETRY EXPERIMENTS COMPLETE                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results summary (experiment -> best test AP):"
echo "--------------------------------------------------------------"
for name in \
    G1b_hop_mode_window_w1 \
    G2a_global_heads_0 G2b_global_heads_2 G2c_global_heads_3 G2d_global_heads_5 \
    G3a_moe_dense G3c_moe_top3 G3d_moe_top5 \
    G4a_multihop_sum G4b_multihop_mean G4c_multihop_sum_no_global G4d_multihop_mean_no_global \
    G5a_cross_hop G5c_cross_hop_embed G5e_cross_hop_full \
    G7a_mask_adj_power \
    G9c_gat_1layer_edge \
    G14b_focal_g1 \
    G16a_h16_d128 G16b_h20_d160 \
    G18e_kitchen_sink; do
    logfile="${LOG_DIR}/${name}.log"
    best=$(grep '^BEST:' "${logfile}" 2>/dev/null || echo "NO RESULT")
    printf "  %-45s %s\n" "${name}" "${best}"
done
echo "--------------------------------------------------------------"
echo ""
echo "Full logs in: ${LOG_DIR}/"
echo "Checkpoints in: ${CKPT_DIR}/"
