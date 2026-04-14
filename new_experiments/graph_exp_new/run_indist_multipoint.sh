#!/usr/bin/env bash
set -eux pipefail

# =============================================================
# Multi-Point & Cross-Attention Ablation Experiments
#
# Experiments:
#   1. gred backbone + cross_attention routing + pma generator
#   2. gred backbone + cross_attention routing + score_based generator
#   3. hybrid backbone + current pipeline + multipoint at [2]
#   4. hybrid backbone + current pipeline + multipoint at [0, 2, 6]
#   5. hybrid backbone + multipoint with cross_attention at [2]
#   6. hybrid backbone + multipoint with cross_attention at [2, 4, 6]
#
# All experiments use batch_size=128.
# Phases 1 & 2 are shared per backbone; Phases 3-5 run per config.
# =============================================================

BASE_DIR="checkpoints_multipoint"
NUM_PROXIES=32
BS=128

# -------------------------------------------------------
# Backbone arg strings (batch_size=128 for all)
# -------------------------------------------------------
VANILLA_ARGS="--backbone vanilla_gt --hidden_dim 256 --state_dim 256 --num_gred_layers 8 --gred_expand 1 --r_min 0.95 --r_max 1.0 --max_phase_lru 6.28 --gred_act full-glu --max_hops 40 --dropout 0.1 --batch_size ${BS} --use_lap_pe"
HYBRID_ARGS="--backbone hybrid --hidden_dim 88 --state_dim 88 --num_gred_layers 8 --num_transformer_layers 2 --num_heads 8 --gred_expand 1 --r_min 0.95 --r_max 1.0 --max_phase_lru 6.28 --gred_act full-glu --max_hops 40 --dropout 0.2 --batch_size 32 --use_lap_pe"

# -------------------------------------------------------
# Generator arg strings
# -------------------------------------------------------
SCORE_ARGS="--generator score_based --gen_num_layers 3 --gen_num_heads 8 --gen_dropout 0.2"
PMA_ARGS="--generator pma --pma_query_mode farthest_point --gen_num_layers 8 --gen_num_heads 8 --gen_dropout 0.2"

# -------------------------------------------------------
# Cross-attention routing args
# -------------------------------------------------------
CROSS_ATTN_ARGS="--use_cross_attn_routing --num_cross_layers 2"


# =============================================================
# Helper: run phases 3-5 for a given backbone+variant combo
# =============================================================
run_generator_phases() {
    local BACKBONE_NAME="$1"
    local BACKBONE_ARGS="$2"
    local EXP_NAME="$3"
    local GEN_ARGS="$4"
    local EXTRA_ARGS="$5"   # e.g. cross-attn flags, insertion layers, etc.
    local P1_MODEL="$6"
    local P2_MODEL="$7"

    local EXP_DIR="${BASE_DIR}/${BACKBONE_NAME}/${EXP_NAME}"
    mkdir -p "${EXP_DIR}"

    echo ""
    echo "============================================================="
    echo "  ${BACKBONE_NAME} | ${EXP_NAME}: Phase 3 (train generator)"
    echo "============================================================="
    python3 main_indist.py \
        --phase 3 \
        ${BACKBONE_ARGS} \
        ${GEN_ARGS} \
        ${EXTRA_ARGS} \
        --num_proxies ${NUM_PROXIES} \
        --model_path "${P1_MODEL}" \
        --phase2_model_path "${P2_MODEL}" \
        --p3_lr 1e-3 \
        --p3_max_epochs 200 \
        --p3_patience 50 \
        --p3_recon_weight 1.0 \
        --p3_recon_anneal_to 0.1 \
        --p3_recon_anneal_epochs 100 \
        --save_dir "${EXP_DIR}"

    local GEN_PATH="${EXP_DIR}/phase3_generator.pt"

    echo ""
    echo "============================================================="
    echo "  ${BACKBONE_NAME} | ${EXP_NAME}: Phase 4 (augmented eval)"
    echo "============================================================="
    python3 main_indist.py \
        --phase 4 \
        ${BACKBONE_ARGS} \
        ${GEN_ARGS} \
        ${EXTRA_ARGS} \
        --num_proxies ${NUM_PROXIES} \
        --model_path "${P1_MODEL}" \
        --generator_path "${GEN_PATH}" \
        --save_dir "${EXP_DIR}"

    echo ""
    echo "============================================================="
    echo "  ${BACKBONE_NAME} | ${EXP_NAME}: Phase 5 (E2E fine-tune)"
    echo "============================================================="
    python3 main_indist.py \
        --phase 5 \
        ${BACKBONE_ARGS} \
        ${GEN_ARGS} \
        ${EXTRA_ARGS} \
        --num_proxies ${NUM_PROXIES} \
        --model_path "${P1_MODEL}" \
        --generator_path "${GEN_PATH}" \
        --p5_phase_a_epochs 20 \
        --p5_max_epochs 200 \
        --p5_patience 50 \
        --p5_proxy_dropout 0.1 \
        --save_dir "${EXP_DIR}"
}


# =============================================================
# A. GRED backbone (for cross-attention experiments only)
#    Note: standalone GRED doesn't natively support proxy concat,
#    but cross-attention routing is applied at the backbone level.
# =============================================================

GR_DIR="${BASE_DIR}/gred"
GR_SHARED="${GR_DIR}/shared"
mkdir -p "${GR_SHARED}"

echo "============================================================="
echo "  gred: Phase 1 (pretrain)"
echo "============================================================="
python3 main_indist.py \
    --phase 1 \
    ${VANILLA_ARGS} \
    ${CROSS_ATTN_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --p1_lr 5e-5 \
    --p1_weight_decay 1e-4 \
    --p1_max_epochs 200 \
    --p1_patience 50 \
    --save_dir "${GR_SHARED}"

GR_P1="${GR_SHARED}/phase1_best.pt"

echo ""
echo "============================================================="
echo "  gred: Phase 2 (partial-graph, N-M nodes)"
echo "============================================================="
python3 main_indist.py \
    --phase 2 \
    ${VANILLA_ARGS} \
    ${CROSS_ATTN_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --model_path "${GR_P1}" \
    --p2_lr 5e-5 \
    --p2_weight_decay 1e-4 \
    --p2_max_epochs 200 \
    --p2_patience 50 \
    --save_dir "${GR_SHARED}"

GR_P2="${GR_SHARED}/phase2_best.pt"

# Exp 1: gred + cross_attention + pma
run_generator_phases "gred" "${VANILLA_ARGS}" "cross_attn_pma" \
    "${PMA_ARGS}" "${CROSS_ATTN_ARGS}" "${GR_P1}" "${GR_P2}"

# Exp 2: gred + cross_attention + score_based
run_generator_phases "gred" "${VANILLA_ARGS}" "cross_attn_score" \
    "${SCORE_ARGS}" "${CROSS_ATTN_ARGS}" "${GR_P1}" "${GR_P2}"


# =============================================================
# B. HYBRID backbone (for multipoint & multipoint+cross-attn)
#    Phases 1 & 2 are shared across all hybrid experiments.
# =============================================================

HY_DIR="${BASE_DIR}/hybrid"
HY_SHARED="${HY_DIR}/shared"
mkdir -p "${HY_SHARED}"

echo ""
echo "============================================================="
echo "  hybrid: Phase 1 (pretrain)"
echo "============================================================="
python3 main_indist.py \
    --phase 1 \
    ${HYBRID_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --p1_lr 5e-5 \
    --p1_weight_decay 1e-4 \
    --p1_max_epochs 200 \
    --p1_patience 50 \
    --save_dir "${HY_SHARED}"

HY_P1="${HY_SHARED}/phase1_best.pt"

echo ""
echo "============================================================="
echo "  hybrid: Phase 2 (partial-graph, N-M nodes)"
echo "============================================================="
python3 main_indist.py \
    --phase 2 \
    ${HYBRID_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --model_path "${HY_P1}" \
    --p2_lr 1e-3 \
    --p2_weight_decay 1e-4 \
    --p2_max_epochs 200 \
    --p2_patience 50 \
    --save_dir "${HY_SHARED}"

HY_P2="${HY_SHARED}/phase2_best.pt"

# Exp 3: hybrid + current pipeline + multipoint at [2]
run_generator_phases "hybrid" "${HYBRID_ARGS}" "multipoint_layer2" \
    "${SCORE_ARGS}" "--proxy_insertion_layers 2" "${HY_P1}" "${HY_P2}"

# Exp 4: hybrid + current pipeline + multipoint at [0, 2, 6]
run_generator_phases "hybrid" "${HYBRID_ARGS}" "multipoint_layers0_2_6" \
    "${SCORE_ARGS}" "--proxy_insertion_layers 0,2,6" "${HY_P1}" "${HY_P2}"

# Exp 5: hybrid + multipoint + cross_attention at [2]
run_generator_phases "hybrid" "${HYBRID_ARGS}" "multipoint_cross_attn_layer2" \
    "${SCORE_ARGS}" "${CROSS_ATTN_ARGS} --proxy_insertion_layers 2" "${HY_P1}" "${HY_P2}"

# Exp 6: hybrid + multipoint + cross_attention at [0, 3, 6]
run_generator_phases "hybrid" "${HYBRID_ARGS}" "multipoint_cross_attn_layers0_3_6" \
    "${SCORE_ARGS}" "${CROSS_ATTN_ARGS} --proxy_insertion_layers 0,3,6" "${HY_P1}" "${HY_P2}"

# Exp 7: hybrid + multipoint + cross_attention at [0, 3, 6]
run_generator_phases "hybrid" "${HYBRID_ARGS}" "multipoint_cross_attn_layers0_3_6_sep_gen" \
    "${SCORE_ARGS}" "${CROSS_ATTN_ARGS} --proxy_insertion_layers 0,3,6 --separate_proxy_generators --separate_proxy_routers" "${HY_P1}" "${HY_P2}"


echo ""
echo "============================================================="
echo "  All multipoint & cross-attention experiments complete."
echo ""
echo "  Results structure:"
echo "    ${BASE_DIR}/"
echo "      gred/"
echo "        shared/                        — phase1_best.pt, phase2_best.pt"
echo "        cross_attn_pma/                — phase3_generator.pt, phase4_results.pkl, phase5_best.pt"
echo "        cross_attn_score/              — ..."
echo "      hybrid/"
echo "        shared/                        — phase1_best.pt, phase2_best.pt"
echo "        multipoint_layer2/             — ..."
echo "        multipoint_layers0_2_6/        — ..."
echo "        multipoint_cross_attn_layer2/  — ..."
echo "        multipoint_cross_attn_layers2_4_6/ — ..."
echo "============================================================="
