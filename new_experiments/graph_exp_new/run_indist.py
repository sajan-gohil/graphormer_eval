#!/usr/bin/env bash
set -eux pipefail

# =============================================================
# In-Distribution Proxy Learning (IDPL) experiments
#
# Tests all backbone × generator combinations through the
# 5-phase pipeline in main_indist.py.
#
# Structure:
#   Phase 1 is trained once per backbone (shared).
#   Phase 2 is trained once per backbone (shared — only depends on backbone).
#   Phase 3–5 run per generator type.
#
# Backbones: vanilla_gt, hybrid
#   (gred standalone excluded — no proxy support)
# Generators: score_based, pma, graph_coarsening, gnn_pooling
# =============================================================

BASE_DIR="checkpoints_indist"
NUM_PROXIES=32

# Common model args per backbone
VANILLA_ARGS="--backbone vanilla_gt --hidden_dim 256 --num_layers 5 --num_heads 8 --dropout 0.1 --batch_size 128"
HYBRID_ARGS="--backbone hybrid --hidden_dim 88 --state_dim 88 --num_gred_layers 8 --num_transformer_layers 2 --num_heads 8 --gred_expand 1 --r_min 0.95 --r_max 1.0 --max_phase_lru 6.28 --gred_act full-glu --max_hops 40 --dropout 0.2 --batch_size 8"

# Generator-specific args
SCORE_ARGS="--generator score_based --gen_num_layers 4 --gen_num_heads 8 --gen_dropout 0.2"
PMA_FP_ARGS="--generator pma --pma_query_mode farthest_point --gen_num_layers 4 --gen_num_heads 8 --gen_dropout 0.2"
PMA_SK_ARGS="--generator pma --pma_query_mode soft_kmeans --gen_num_layers 4 --gen_num_heads 8 --gen_dropout 0.2"
COARSEN_ARGS="--generator graph_coarsening --gen_num_layers 4 --gen_num_heads 8 --gen_dropout 0.2 --coarsen_gnn_type GIN --coarsen_reg_weight 0.1"
GNN_POOL_ARGS="--generator gnn_pooling --gnn_layers 4 --gnn_type GINE --pool_types max --decode_hidden 128 --decode_layers 3 --gen_dropout 0.2"


# =============================================================
# Helper: run phases 3-5 for a given backbone+generator combo
# =============================================================
run_generator_phases() {
    local BACKBONE_NAME="$1"
    local BACKBONE_ARGS="$2"
    local GEN_NAME="$3"
    local GEN_ARGS="$4"
    local P1_MODEL="$5"
    local P2_MODEL="$6"

    local EXP_DIR="${BASE_DIR}/${BACKBONE_NAME}/${GEN_NAME}"
    mkdir -p "${EXP_DIR}"

    echo ""
    echo "============================================================="
    echo "  ${BACKBONE_NAME} + ${GEN_NAME}: Phase 3 (train generator)"
    echo "============================================================="
    python3 main_indist.py \
        --phase 3 \
        ${BACKBONE_ARGS} \
        ${GEN_ARGS} \
        --num_proxies ${NUM_PROXIES} \
        --model_path "${P1_MODEL}" \
        --phase2_model_path "${P2_MODEL}" \
        --p3_lr 1e-3 \
        --p3_max_epochs 1000 \
        --p3_patience 50 \
        --p3_recon_weight 1.0 \
        --p3_recon_anneal_to 0.1 \
        --p3_recon_anneal_epochs 100 \
        --save_dir "${EXP_DIR}"

    local GEN_PATH="${EXP_DIR}/phase3_generator.pt"

    echo ""
    echo "============================================================="
    echo "  ${BACKBONE_NAME} + ${GEN_NAME}: Phase 4 (augmented eval)"
    echo "============================================================="
    python3 main_indist.py \
        --phase 4 \
        ${BACKBONE_ARGS} \
        ${GEN_ARGS} \
        --num_proxies ${NUM_PROXIES} \
        --model_path "${P1_MODEL}" \
        --generator_path "${GEN_PATH}" \
        --save_dir "${EXP_DIR}"

    echo ""
    echo "============================================================="
    echo "  ${BACKBONE_NAME} + ${GEN_NAME}: Phase 5 (E2E fine-tune)"
    echo "============================================================="
    python3 main_indist.py \
        --phase 5 \
        ${BACKBONE_ARGS} \
        ${GEN_ARGS} \
        --num_proxies ${NUM_PROXIES} \
        --model_path "${P1_MODEL}" \
        --generator_path "${GEN_PATH}" \
        --p5_phase_a_epochs 20 \
        --p5_max_epochs 500 \
        --p5_patience 40 \
        --p5_proxy_dropout 0.1 \
        --save_dir "${EXP_DIR}"
}


# =============================================================
# A. VANILLA TRANSFORMER BACKBONE
# =============================================================

VT_DIR="${BASE_DIR}/vanilla_gt"
VT_SHARED="${VT_DIR}/shared"
mkdir -p "${VT_SHARED}"

echo "============================================================="
echo "  vanilla_gt: Phase 1 (pretrain full transformer)"
echo "============================================================="
python3 main_indist.py \
    --phase 1 \
    ${VANILLA_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --p1_lr 1e-3 \
    --p1_max_epochs 500 \
    --p1_patience 50 \
    --save_dir "${VT_SHARED}"

VT_P1="${VT_SHARED}/phase1_best.pt"

echo ""
echo "============================================================="
echo "  vanilla_gt: Phase 2 (partial-graph, N-M nodes)"
echo "============================================================="
python3 main_indist.py \
    --phase 2 \
    ${VANILLA_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --model_path "${VT_P1}" \
    --p2_lr 1e-3 \
    --p2_max_epochs 500 \
    --p2_patience 50 \
    --save_dir "${VT_SHARED}"

VT_P2="${VT_SHARED}/phase2_best.pt"

# --- vanilla_gt × all generators ---
run_generator_phases "vanilla_gt" "${VANILLA_ARGS}" "score_based"      "${SCORE_ARGS}"   "${VT_P1}" "${VT_P2}"
run_generator_phases "vanilla_gt" "${VANILLA_ARGS}" "pma_fp"           "${PMA_FP_ARGS}"  "${VT_P1}" "${VT_P2}"
run_generator_phases "vanilla_gt" "${VANILLA_ARGS}" "pma_sk"           "${PMA_SK_ARGS}"  "${VT_P1}" "${VT_P2}"
run_generator_phases "vanilla_gt" "${VANILLA_ARGS}" "graph_coarsening" "${COARSEN_ARGS}" "${VT_P1}" "${VT_P2}"
run_generator_phases "vanilla_gt" "${VANILLA_ARGS}" "gnn_pooling"      "${GNN_POOL_ARGS}" "${VT_P1}" "${VT_P2}"


# =============================================================
# B. HYBRID (GRED + TRANSFORMER) BACKBONE
# =============================================================

HY_DIR="${BASE_DIR}/hybrid"
HY_SHARED="${HY_DIR}/shared"
mkdir -p "${HY_SHARED}"

echo ""
echo "============================================================="
echo "  hybrid: Phase 1 (pretrain full model)"
echo "============================================================="
python3 main_indist.py \
    --phase 1 \
    ${HYBRID_ARGS} \
    --num_proxies ${NUM_PROXIES} \
    --p1_lr 1e-3 \
    --p1_weight_decay 3e-4 \
    --p1_max_epochs 500 \
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
    --p2_weight_decay 3e-4 \
    --p2_max_epochs 500 \
    --p2_patience 50 \
    --save_dir "${HY_SHARED}"

HY_P2="${HY_SHARED}/phase2_best.pt"

# --- hybrid × all generators ---
run_generator_phases "hybrid" "${HYBRID_ARGS}" "score_based"      "${SCORE_ARGS}"   "${HY_P1}" "${HY_P2}"
run_generator_phases "hybrid" "${HYBRID_ARGS}" "pma_fp"           "${PMA_FP_ARGS}"  "${HY_P1}" "${HY_P2}"
run_generator_phases "hybrid" "${HYBRID_ARGS}" "pma_sk"           "${PMA_SK_ARGS}"  "${HY_P1}" "${HY_P2}"
run_generator_phases "hybrid" "${HYBRID_ARGS}" "graph_coarsening" "${COARSEN_ARGS}" "${HY_P1}" "${HY_P2}"
run_generator_phases "hybrid" "${HYBRID_ARGS}" "gnn_pooling"      "${GNN_POOL_ARGS}" "${HY_P1}" "${HY_P2}"


echo ""
echo "============================================================="
echo "  All IDPL experiments complete."
echo ""
echo "  Results structure:"
echo "    ${BASE_DIR}/"
echo "      vanilla_gt/"
echo "        shared/        — phase1_best.pt, phase2_best.pt"
echo "        score_based/   — phase3_generator.pt, phase4_results.pkl, phase5_best.pt"
echo "        pma_fp/        — ..."
echo "        pma_sk/        — ..."
echo "        graph_coarsening/ — ..."
echo "        gnn_pooling/   — ..."
echo "      hybrid/"
echo "        shared/        — phase1_best.pt, phase2_best.pt"
echo "        score_based/   — ..."
echo "        pma_fp/        — ..."
echo "        pma_sk/        — ..."
echo "        graph_coarsening/ — ..."
echo "        gnn_pooling/   — ..."
echo "============================================================="
