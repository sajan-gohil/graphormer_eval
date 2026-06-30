#!/bin/bash
# =============================================================================
# Ablation / Component Sweep for Hop-Masked Transformer on Peptides-func
# =============================================================================
#
# Best known baseline config:
#   max_hops=30  num_heads=30  num_global_heads=1  hop_mode=single
#   hop_window=0  hidden_dim=240  (8 per head)
#
# This script runs a systematic sweep of individual components and modes to
# identify which give a lift in AP.  Each experiment logs to its own file.
#
# Usage:
#   chmod +x run_ablation_experiments.sh
#   bash run_ablation_experiments.sh
#
# To resume after a crash, grep for "DONE" in the log dir and skip completed.
# =============================================================================

set -e  # exit on error (remove if you want all experiments to try running)

# ─── Paths & common settings ─────────────────────────────────────────────────
SCRIPT="train_hop_masked_transformer_final.py"
DATASET="Peptides-func"
LOG_DIR="ablation_logs"
CKPT_DIR="ablation_checkpoints"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

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
  --seed ${BASE_SEED}"

# Helper: run an experiment
# Args: $1 = experiment name, $2 = extra flags (space-separated)
run_exp() {
    local name="$1"
    shift
    local extra="$@"
    local logfile="${LOG_DIR}/${name}.log"

    echo "============================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  LOG: ${logfile}"
    echo "  FLAGS: ${extra}"
    echo "============================================================"

    python3 ${SCRIPT} ${COMMON} \
        --save_dir "${CKPT_DIR}/${name}" \
        ${extra} \
        > "${logfile}" 2>&1

    # Print the BEST line for quick comparison
    echo "  >> $(grep '^BEST:' ${logfile} || echo 'NO BEST LINE FOUND')"
    echo ""
}

# =============================================================================
# GROUP 0: BASELINE (best known config, exact reproduction)
# =============================================================================
echo "===== GROUP 0: BASELINE ====="

run_exp "G0_baseline_best" \
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
run_exp "G1a_hop_mode_contiguous" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode contiguous \
    --hop_window ${BASE_HOP_WINDOW}

# 1b. Window with half-width=1 (each head sees ~3 adjacent hops)
run_exp "G1b_hop_mode_window_w1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode window \
    --hop_window 1

# 1c. Window with half-width=2 (each head sees ~5 adjacent hops)
run_exp "G1c_hop_mode_window_w2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode window \
    --hop_window 2

# 1d. Interleaved
run_exp "G1d_hop_mode_interleaved" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode interleaved \
    --hop_window 1

# 1e. Alternating (even layers get even hops, odd layers get odd hops)
run_exp "G1e_hop_mode_alternating" \
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

run_exp "G2a_global_heads_0" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads 0 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

run_exp "G2b_global_heads_2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads 2 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

run_exp "G2c_global_heads_3" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads 3 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

run_exp "G2d_global_heads_5" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads 5 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# =============================================================================
# GROUP 3: MoE GATING (learned hop assignment)
#   Replaces deterministic hop-to-head mapping with learned gating.
# =============================================================================
echo "===== GROUP 3: MoE GATING ====="

# 3a. MoE dense (full softmax over all K hops)
run_exp "G3a_moe_dense" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 0

# 3b. MoE sparse top-1
run_exp "G3b_moe_top1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 1

# 3c. MoE sparse top-3
run_exp "G3c_moe_top3" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_moe_gating \
    --top_k 3

# 3d. MoE sparse top-5
run_exp "G3d_moe_top5" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
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
run_exp "G4a_multihop_sum" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout sum

# 4b. Multihop mean readout + global view
run_exp "G4b_multihop_mean" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout mean

# 4c. Multihop sum readout, NO global view
run_exp "G4c_multihop_sum_no_global" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --multihop_attn \
    --multihop_readout sum \
    --multihop_no_global

# 4d. Multihop mean readout, NO global view
run_exp "G4d_multihop_mean_no_global" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
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
run_exp "G5a_cross_hop" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop

# 5b. Cross-hop mixer + block_diag_out (recommended pairing)
run_exp "G5b_cross_hop_block_diag" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --block_diag_out

# 5c. Cross-hop mixer + hop embedding
run_exp "G5c_cross_hop_embed" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --cross_hop_hop_embedding

# 5d. Cross-hop mixer + no-FFN mode
run_exp "G5d_cross_hop_no_ffn" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --cross_hop_no_ffn

# 5e. Cross-hop + block_diag + hop_embed + no_ffn (all cross-hop features)
run_exp "G5e_cross_hop_full" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
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
run_exp "G6a_blend_adj_power" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power

# 6b. Edge feature attention bias (hop-1 heads)
run_exp "G6b_edge_bias" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_bias

# 6c. RRWP bias (dim=8)
run_exp "G6c_rrwp_d8" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_rrwp \
    --rrwp_dim 8

# 6d. RRWP bias (dim=16)
run_exp "G6d_rrwp_d16" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_rrwp \
    --rrwp_dim 16

# 6e. Adj-power blend + edge bias
run_exp "G6e_blend_adj_edge_bias" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power \
    --use_edge_bias

# 6f. Adj-power blend + RRWP
run_exp "G6f_blend_adj_rrwp" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --blend_adj_power \
    --use_rrwp \
    --rrwp_dim 8

# 6g. All attention biases combined
run_exp "G6g_all_attn_biases" \
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
run_exp "G7a_mask_adj_power" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --mask_type adj_power

# 7b. adj_power with self loops  (A+I)^k
run_exp "G7b_mask_adj_power_self_loops" \
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
run_exp "G8a_edge_features" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features

# 8b. Laplacian PE
run_exp "G8b_lap_pe" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_lap_pe \
    --lap_pe_dim 8

# 8c. Edge features + Lap PE
run_exp "G8c_edge_lap" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_lap_pe \
    --lap_pe_dim 8

# 8d. Virtual node
run_exp "G8d_virtual_node" \
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
run_exp "G9a_gat_1layer" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 4

# 9b. 2 GATv2 layers (4 heads)
run_exp "G9b_gat_2layer" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 2 \
    --num_gat_heads 4

# 9c. 1 GATv2 layer + edge features
run_exp "G9c_gat_1layer_edge" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_post_gat_layers 1 \
    --num_gat_heads 4 \
    --use_edge_features

# =============================================================================
# GROUP 10: NORMALIZATION TYPE
# =============================================================================
echo "===== GROUP 10: NORM TYPE ====="

# 10a. RMSNorm
run_exp "G10a_norm_rms" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --norm_type rms

# 10b. GraphNorm
run_exp "G10b_norm_graph" \
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
run_exp "G11a_pool_mean" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --graph_pool mean

# 11b. Attention pooling (Set-Transformer PMA)
run_exp "G11b_pool_attention" \
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
run_exp "G12a_block_diag_out" \
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
run_exp "G13a_v_head_dim_16" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --v_head_dim 16

# 13b. V head dim = 32
run_exp "G13b_v_head_dim_32" \
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
run_exp "G14a_pos_weight" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_pos_weight

# 14b. Focal loss (gamma=1)
run_exp "G14b_focal_g1" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --focal_gamma 1.0

# 14c. Focal loss (gamma=2)
run_exp "G14c_focal_g2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --focal_gamma 2.0

# 14d. Label smoothing (eps=0.05)
run_exp "G14d_label_smooth_005" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --label_smoothing 0.05

# 14e. Label smoothing (eps=0.1)
run_exp "G14e_label_smooth_01" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --label_smoothing 0.1

# 14f. Pos weight + focal (gamma=1)
run_exp "G14f_pos_weight_focal" \
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
run_exp "G15a_layers_2" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_layers 2

# 15b. 6 layers
run_exp "G15b_layers_6" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --num_layers 6

# 15c. 8 layers
run_exp "G15c_layers_8" \
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
run_exp "G16a_h16_d128" \
    --hidden_dim 128 \
    --num_heads 16 \
    --num_global_heads 1 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# 16b. 20 heads × 8 = hidden_dim 160
run_exp "G16b_h20_d160" \
    --hidden_dim 160 \
    --num_heads 20 \
    --num_global_heads 1 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# 16c. 30 heads × 12 = hidden_dim 360 (bigger per-head dim)
run_exp "G16c_h30_d360" \
    --hidden_dim 360 \
    --num_heads 30 \
    --num_global_heads 1 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# 16d. 30 heads × 16 = hidden_dim 480 (even bigger)
run_exp "G16d_h30_d480" \
    --hidden_dim 480 \
    --num_heads 30 \
    --num_global_heads 1 \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW}

# =============================================================================
# GROUP 17: DROPOUT ABLATION
# =============================================================================
echo "===== GROUP 17: DROPOUT ====="

run_exp "G17a_dropout_01" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dropout 0.1

run_exp "G17b_dropout_03" \
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
run_exp "G18a_edge_edge_bias_rrwp" \
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
run_exp "G18b_edge_vnode_lap" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_virtual_node \
    --use_lap_pe \
    --lap_pe_dim 8

# 18c. Cross-hop + blend adj + edge features
run_exp "G18c_cross_hop_blend_edge" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --dynamic_cross_hop \
    --blend_adj_power \
    --use_edge_features

# 18d. GATv2 + edge features + label smoothing
run_exp "G18d_gat_edge_ls" \
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
run_exp "G18e_kitchen_sink" \
    --hidden_dim ${BASE_HIDDEN_DIM} \
    --num_heads ${BASE_NUM_HEADS} \
    --num_global_heads ${BASE_NUM_GLOBAL_HEADS} \
    --hop_mode ${BASE_HOP_MODE} \
    --hop_window ${BASE_HOP_WINDOW} \
    --use_edge_features \
    --use_lap_pe --lap_pe_dim 8 \
    --use_rrwp --rrwp_dim 8 \
    --blend_adj_power \
    --use_virtual_node

# =============================================================================
# SUMMARY EXTRACTION
# =============================================================================
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results summary (experiment → best test AP):"
echo "--------------------------------------------------------------"
for logfile in ${LOG_DIR}/*.log; do
    name=$(basename "${logfile}" .log)
    best=$(grep '^BEST:' "${logfile}" 2>/dev/null || echo "NO RESULT")
    printf "  %-45s %s\n" "${name}" "${best}"
done
echo "--------------------------------------------------------------"
echo ""
echo "Full logs in: ${LOG_DIR}/"
echo "Checkpoints in: ${CKPT_DIR}/"
