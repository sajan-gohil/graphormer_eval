#!/bin/bash
# =============================================================================
# Ablation / Component Sweep for Hop-Masked Transformer on Peptides-func
# =============================================================================
#
# Best known baseline config:
#   max_hops=30  num_heads=30  num_global_heads=1  hop_mode=single
#   hop_window=0  hidden_dim=240  (8 per head)
#
# Fix applied from retry script: for single hop_mode,
#   num_heads = num_global_heads + (max_hops - 1)
# For G16a/G16b (smaller width), max_hops is reduced to satisfy:
#   max_hops = num_heads - num_global_heads + 1
#
# This script runs a systematic sweep of individual components and modes to
# identify which give a lift in AP.  Each experiment logs to its own file.
#
# Runs 6 experiments in parallel: 3 on cuda:1 + 3 on cuda:2.
# A new batch of 6 starts once the previous batch finishes.
#
# Usage:
#   chmod +x run_ablation_experiments.sh
#   bash run_ablation_experiments.sh
#
# To resume after a crash, grep for "DONE" in the log dir and skip completed.
# =============================================================================

# ─── Paths & common settings ─────────────────────────────────────────────────
SCRIPT="train_hop_masked_transformer_final.py"
DATASET="Peptides-func"
LOG_DIR="ablation_logs"
CKPT_DIR="ablation_checkpoints"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# GPUs to use (3 jobs per GPU, 6 total in parallel)
GPUS=("cuda:1" "cuda:2")
JOBS_PER_GPU=3

# Best-known base config
BASE_MAX_HOPS=30
BASE_NUM_HEADS=30
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

# Shared flags that stay constant across experiments
# NOTE: --num_workers 0 to avoid deadlocks (fork + numpy/scipy BLAS locks +
# CUDA context inheritance causes workers to hang at 0% CPU while holding GPU
# memory). The collate_fn does heavy numpy ops (B,K,N,N) tensors and
# torch_geometric's Batch.from_data_list can also trigger threading issues
# under forked workers. Setting to 0 is the safe default.
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

# ─── Parallel execution machinery ────────────────────────────────────────────
# We collect experiments into a queue, then dispatch them in batches of 6
# (3 per GPU). Each batch runs fully in parallel; the next batch starts only
# after all jobs in the current batch finish.

# Queue: each entry is "name|extra_flags"
QUEUE=()

enqueue() {
    # Args: $1 = experiment name, rest = extra flags
    local name="$1"
    shift
    QUEUE+=("${name}|$*")
}

flush_queue() {
    # Dispatch everything in QUEUE in batches of JOBS_PER_GPU * len(GPUS).
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
        echo "║  (experiments $((start+1))–${end} of ${total})"
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

            # Skip if already completed
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
                >> "${logfile}" 2>&1 &

            pids+=($!)
            names+=("${name}")

            # Round-robin GPU assignment: 3 jobs per GPU
            gpu_slot=$((gpu_slot + 1))
            if (( gpu_slot >= JOBS_PER_GPU )); then
                gpu_slot=0
                gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
            fi
        done

        # Wait for all jobs in this batch
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
                    echo "  ✗ FAILED (rc=${rc}): ${n}"
                    failed=$((failed + 1))
                else
                    local best=$(grep '^BEST:' "${lf}" 2>/dev/null || echo "NO BEST LINE")
                    echo "  ✓ DONE: ${n}  >>  ${best}"
                fi
            done
            echo "  Batch ${batch_num} complete (${#pids[@]} ran, ${failed} failed)."
        fi
    done

    # Clear the queue
    QUEUE=()
}

# =============================================================================
# GROUP 0: BASELINE (best known config, exact reproduction)
# =============================================================================
echo "===== GROUP 0: BASELINE ====="

enqueue "G0_baseline_best" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# =============================================================================
# GROUP 1: HOP MODE ABLATION
#   Keep heads=30, global=1, hidden=240. Vary how hops are assigned to heads.
# =============================================================================
echo "===== GROUP 1: HOP MODE ABLATION ====="

# 1a. Contiguous (partition [1..K-1] into chunks per head)
enqueue "G1a_hop_mode_contiguous" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode contiguous \
    --hop_window ${BASE_HOP_WINDOW}

# 1b. Window with half-width=1 (each head sees ~3 adjacent hops)
enqueue "G1b_hop_mode_window_w1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode window \
    --hop_window 1

# 1c. Window with half-width=2 (each head sees ~5 adjacent hops)
enqueue "G1c_hop_mode_window_w2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode window \
    --hop_window 2

# 1d. Interleaved
enqueue "G1d_hop_mode_interleaved" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode interleaved \
    --hop_window 1

# 1e. Alternating (even layers get even hops, odd layers get odd hops)
enqueue "G1e_hop_mode_alternating" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode alternating \
    --hop_window ${BASE_HOP_WINDOW}

# =============================================================================
# GROUP 2: GLOBAL HEADS ABLATION
#   Baseline is 1 global head. How many free-attention heads help?
# =============================================================================
echo "===== GROUP 2: GLOBAL HEADS ABLATION ====="

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

# Flush baseline + hop modes + global heads (12 experiments = 2 batches of 6)
flush_queue

# =============================================================================
# GROUP 3: MoE GATING (learned hop assignment)
#   Replaces deterministic hop-to-head mapping with learned gating.
# =============================================================================
echo "===== GROUP 3: MoE GATING ====="

# 3a. MoE dense (full softmax over all K hops)
enqueue "G3a_moe_dense" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 0

# 3b. MoE sparse top-1
enqueue "G3b_moe_top1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 1

# 3c. MoE sparse top-3
enqueue "G3c_moe_top3" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 3

# 3d. MoE sparse top-5
enqueue "G3d_moe_top5" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 5

# =============================================================================
# GROUP 4: MULTIHOP ATTENTION
#   Every head sees every hop (H×K views), collapsed by readout.
# =============================================================================
echo "===== GROUP 4: MULTIHOP ATTENTION ====="

# 4a. Multihop sum readout + global view
enqueue "G4a_multihop_sum" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout sum

# 4b. Multihop mean readout + global view
enqueue "G4b_multihop_mean" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout mean

# Flush MoE + first 2 multihop (6 experiments)
flush_queue

# 4c. Multihop sum readout, NO global view
enqueue "G4c_multihop_sum_no_global" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout sum \
    --multihop_no_global

# 4d. Multihop mean readout, NO global view
enqueue "G4d_multihop_mean_no_global" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout mean \
    --multihop_no_global

# =============================================================================
# GROUP 5: DYNAMIC CROSS-HOP MIXER
#   Cross-hop attention sublayer between MHA and FFN.
# =============================================================================
echo "===== GROUP 5: DYNAMIC CROSS-HOP MIXER ====="

# 5a. Cross-hop mixer (with FFN)
enqueue "G5a_cross_hop" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop

# 5b. Cross-hop mixer + block_diag_out (recommended pairing)
enqueue "G5b_cross_hop_block_diag" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --block_diag_out

# 5c. Cross-hop mixer + hop embedding
enqueue "G5c_cross_hop_embed" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --cross_hop_hop_embedding

# 5d. Cross-hop mixer + no-FFN mode
enqueue "G5d_cross_hop_no_ffn" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --cross_hop_no_ffn

# Flush multihop remainder + cross-hop (6 experiments)
flush_queue

# 5e. Cross-hop + block_diag + hop_embed + no_ffn (all cross-hop features)
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

# =============================================================================
# GROUP 6: ATTENTION BIASES (blend_adj_power, edge_bias, RRWP)
# =============================================================================
echo "===== GROUP 6: ATTENTION BIASES ====="

# 6a. Adj-power blend (learnable per-head gamma)
enqueue "G6a_blend_adj_power" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power

# 6b. Edge feature attention bias (hop-1 heads)
enqueue "G6b_edge_bias" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_bias

# 6c. RRWP bias (dim=8)
enqueue "G6c_rrwp_d8" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_rrwp \
    --rrwp_dim 8

# 6d. RRWP bias (dim=16)
enqueue "G6d_rrwp_d16" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_rrwp \
    --rrwp_dim 16

# 6e. Adj-power blend + edge bias
enqueue "G6e_blend_adj_edge_bias" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power \
    --use_edge_bias

# Flush cross-hop-full + attention biases (6 experiments)
flush_queue

# 6f. Adj-power blend + RRWP
enqueue "G6f_blend_adj_rrwp" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power \
    --use_rrwp \
    --rrwp_dim 8

# 6g. All attention biases combined
enqueue "G6g_all_attn_biases" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power \
    --use_edge_bias \
    --use_rrwp \
    --rrwp_dim 8

# =============================================================================
# GROUP 7: MASK TYPE ABLATION (shortest_path vs adj_power)
# =============================================================================
echo "===== GROUP 7: MASK TYPE ABLATION ====="

# 7a. adj_power (A^k)
enqueue "G7a_mask_adj_power" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --mask_type adj_power

# 7b. adj_power with self loops  (A+I)^k
enqueue "G7b_mask_adj_power_self_loops" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --mask_type adj_power \
    --adj_self_loops

# =============================================================================
# GROUP 8: EDGE & STRUCTURAL FEATURES
# =============================================================================
echo "===== GROUP 8: EDGE & STRUCTURAL FEATURES ====="

# 8a. Edge features (BondEncoder in node encoder)
enqueue "G8a_edge_features" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features

# 8b. Laplacian PE
enqueue "G8b_lap_pe" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_lap_pe \
    --lap_pe_dim 8

# Flush biases remainder + mask types + edge/struct (6 experiments)
flush_queue

# 8c. Edge features + Lap PE
enqueue "G8c_edge_lap" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_lap_pe \
    --lap_pe_dim 8

# 8d. Virtual node
enqueue "G8d_virtual_node" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_virtual_node

# =============================================================================
# GROUP 9: POST-TRANSFORMER GATv2 LAYERS
# =============================================================================
echo "===== GROUP 9: POST-TRANSFORMER GATv2 ====="

# 9a. 1 GATv2 layer (4 heads)
enqueue "G9a_gat_1layer" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 4

# 9b. 2 GATv2 layers (4 heads)
enqueue "G9b_gat_2layer" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 2 \
    --num_gat_heads 4

# 9c. 1 GATv2 layer + edge features
enqueue "G9c_gat_1layer_edge" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 4 \
    --use_edge_features

# 9d (bonus). 1 GATv2 layer (8 heads)
enqueue "G9d_gat_1layer_8h" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 8

# Flush edge/struct remainder + GAT (6 experiments)
flush_queue

# =============================================================================
# GROUP 10: NORMALIZATION TYPE
# =============================================================================
echo "===== GROUP 10: NORM TYPE ====="

# 10a. RMSNorm
enqueue "G10a_norm_rms" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --norm_type rms

# 10b. GraphNorm
enqueue "G10b_norm_graph" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --norm_type graph

# =============================================================================
# GROUP 11: GRAPH POOLING
# =============================================================================
echo "===== GROUP 11: GRAPH POOLING ====="

# 11a. Mean pooling
enqueue "G11a_pool_mean" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --graph_pool mean

# 11b. Attention pooling (Set-Transformer PMA)
enqueue "G11b_pool_attention" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --graph_pool attention

# =============================================================================
# GROUP 12: OUTPUT PROJECTION
# =============================================================================
echo "===== GROUP 12: OUTPUT PROJECTION ====="

# 12a. Block-diagonal out_proj (no cross-head mixing inside attention)
enqueue "G12a_block_diag_out" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --block_diag_out

# =============================================================================
# GROUP 13: VALUE HEAD DIM (asymmetric attention)
# =============================================================================
echo "===== GROUP 13: VALUE HEAD DIM ====="

# Base QK head dim = 240/30 = 8.  Try larger V heads.
# 13a. V head dim = 16
enqueue "G13a_v_head_dim_16" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --v_head_dim 16

# Flush norm + pool + block_diag + v_head (6 experiments)
flush_queue

# 13b. V head dim = 32
enqueue "G13b_v_head_dim_32" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --v_head_dim 32

# =============================================================================
# GROUP 14: LOSS FUNCTION VARIANTS
# =============================================================================
echo "===== GROUP 14: LOSS FUNCTION ====="

# 14a. Pos weight (per-class rebalancing)
enqueue "G14a_pos_weight" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_pos_weight

# 14b. Focal loss (gamma=1)
enqueue "G14b_focal_g1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads 30 \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --focal_gamma 1.0

# 14c. Focal loss (gamma=2)
enqueue "G14c_focal_g2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --focal_gamma 2.0

# 14d. Label smoothing (eps=0.05)
enqueue "G14d_label_smooth_005" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --label_smoothing 0.05

# 14e. Label smoothing (eps=0.1)
enqueue "G14e_label_smooth_01" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --label_smoothing 0.1

# Flush v_head_32 + loss variants (6 experiments)
flush_queue

# 14f. Pos weight + focal (gamma=1)
enqueue "G14f_pos_weight_focal" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_pos_weight \
    --focal_gamma 1.0

# =============================================================================
# GROUP 15: DEPTH ABLATION (num_layers)
# =============================================================================
echo "===== GROUP 15: DEPTH ====="

# 15a. 2 layers
enqueue "G15a_layers_2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_layers 2

# 15b. 6 layers
enqueue "G15b_layers_6" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_layers 6

# 15c. 8 layers
enqueue "G15c_layers_8" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_layers 8

# =============================================================================
# GROUP 16: WIDTH ABLATION (hidden_dim / num_heads ratios)
#   Keep 8 per head but vary total count, or keep 30 heads and vary dim/head.
# =============================================================================
echo "===== GROUP 16: WIDTH ====="

# 16a. 16 heads × 8 = hidden_dim 128
enqueue "G16a_h16_d128" \
    --hidden_dim 128 \
    --num_heads 16 \
    --num_global_heads 1 \
    --max_hops 16 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# 16b. 20 heads × 8 = hidden_dim 160
enqueue "G16b_h20_d160" \
    --hidden_dim 160 \
    --num_heads 20 \
    --num_global_heads 1 \
    --max_hops 20 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# Flush loss remainder + depth + width (6 experiments)
flush_queue

# 16c. 30 heads × 12 = hidden_dim 360 (bigger per-head dim)
enqueue "G16c_h30_d360" \
    --hidden_dim 360 \
    --num_heads 30 \
    --num_global_heads 1 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# 16d. 30 heads × 16 = hidden_dim 480 (even bigger)
enqueue "G16d_h30_d480" \
    --hidden_dim 480 \
    --num_heads 30 \
    --num_global_heads 1 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# =============================================================================
# GROUP 17: DROPOUT ABLATION
# =============================================================================
echo "===== GROUP 17: DROPOUT ====="

enqueue "G17a_dropout_01" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dropout 0.1

enqueue "G17b_dropout_03" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dropout 0.3

# =============================================================================
# GROUP 18: BEST COMBO CANDIDATES
#   Combine features that individually show promise (fill in after above runs).
#   These are speculative combos of features from different groups.
# =============================================================================
echo "===== GROUP 18: COMBO CANDIDATES ====="

# 18a. Edge features + edge bias + RRWP
enqueue "G18a_edge_edge_bias_rrwp" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_edge_bias \
    --use_rrwp \
    --rrwp_dim 8

# 18b. Edge features + virtual node + lap PE
enqueue "G18b_edge_vnode_lap" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_virtual_node \
    --use_lap_pe \
    --lap_pe_dim 8

# Flush width remainder + dropout + combo (6 experiments)
flush_queue

# 18c. Cross-hop + blend adj + edge features
enqueue "G18c_cross_hop_blend_edge" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --blend_adj_power \
    --use_edge_features

# 18d. GATv2 + edge features + label smoothing
enqueue "G18d_gat_edge_ls" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 4 \
    --use_edge_features \
    --label_smoothing 0.05

# 18e. Kitchen sink: edge + lap PE + RRWP + blend adj + virtual node
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

# Final flush (3 remaining experiments)
flush_queue

# =============================================================================
# SUMMARY EXTRACTION
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                ALL EXPERIMENTS COMPLETE                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results summary (experiment → best test AP):"
echo "--------------------------------------------------------------"
for logfile in ${LOG_DIR}/*.log; do
    name=$(basename "${logfile}" .log)
    best=$(grep '^BEST:' "${logfile}" 2>/dev/null || echo "NO RESULT")
    printf "  %-45s %s\n" "${name}" "${best}"
done | sort
echo "--------------------------------------------------------------"
echo ""
echo "Full logs in: ${LOG_DIR}/"
echo "Checkpoints in: ${CKPT_DIR}/"