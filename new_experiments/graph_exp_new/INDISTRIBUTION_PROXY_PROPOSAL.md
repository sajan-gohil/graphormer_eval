# In-Distribution Proxy Learning (IDPL) Proposal

## Motivation

Current proxy generation approaches train a generator to produce M virtual nodes that are concatenated alongside the original N nodes before passing through the transformer. The generator must learn to synthesize useful proxy tokens from scratch — a challenging task because the proxy embedding space is unconstrained and potentially out-of-distribution relative to what the transformer has learned to process.

**Key insight:** Instead of generating arbitrary proxy embeddings, we can train a generator that produces *additional node-like embeddings* — tokens that look like real graph nodes to the transformer. By training the generator to produce nodes that are in-distribution w.r.t. the transformer's learned representations, the transformer can leverage them more effectively without requiring extensive co-adaptation.

This proposal introduces a **5-phase In-Distribution Proxy Learning (IDPL)** pipeline that:
1. First establishes a strong transformer baseline on N nodes.
2. Then trains a *partial-graph* transformer on a subset of N−M nodes, establishing a performance gap.
3. Trains a proxy generator to "fill in" the missing M nodes given only the N−M subset.
4. At evaluation, generates M *extra* nodes from the full N-node graph, giving the transformer N+M in-distribution tokens.
5. Fine-tunes the full pipeline end-to-end.

---

## Pipeline Overview

```
Phase 1: Train full transformer on N nodes
    → Establishes baseline AP and learned representations

Phase 2: Train partial transformer on (N-M) node subsets
    → Encoder frozen from Phase 1; only transformer layers retrained
    → Creates a measurable performance gap vs Phase 1

Phase 3: Train proxy generator to reconstruct missing M nodes
    → Input: (N-M) node embeddings from frozen encoder
    → Target: the M held-out node embeddings (from Phase 1 encoder)
    → Validation: passing (N-M) + generated M through frozen transformer
      should recover (some of) the Phase 1 → Phase 2 performance gap

Phase 4: Evaluation with augmented graphs
    → Take original N nodes, generate M additional nodes from proxy generator
    → Pass N+M nodes through transformer
    → Should exceed Phase 1 baseline (more in-distribution information)

Phase 5: End-to-end fine-tuning
    → Unfreeze full pipeline (transformer + generator)
    → Fine-tune jointly with proxy dropout for robustness
```

---

## Detailed Phase Descriptions

### Phase 1: Full Transformer Pretraining

**Objective:** Train a standard graph transformer on the complete graph (all N nodes per graph).

**Setup:**
- Model: `GraphTransformer` / `GREDHybridTransformer` (configurable backbone)
- Training: Standard BCE loss on Peptides-func multi-label classification
- No proxies, no generator — pure baseline

**Output:**
- `phase1_best.pt`: Best model checkpoint (by validation AP)
- Baseline AP on val/test sets

**Why:** Establishes (a) the target representation space for proxy generation, and (b) the performance ceiling that partial-graph training will fall short of.

---

### Phase 2: Partial-Graph Transformer Training

**Objective:** Retrain transformer layers to work with only a random subset of N−M nodes per graph.

**Setup:**
- Load Phase 1 model; **freeze the node encoder** (preserves learned atom/bond embeddings)
- At each training step, for each graph in the batch:
  - Randomly select N−M nodes (where M = `num_proxies`)
  - If N ≤ M, keep all nodes (skip subsampling for tiny graphs)
  - Create a subgraph mask selecting which nodes to keep
- Only the transformer layers and classification head are trained
- Node encoder stays frozen so the embedding space remains consistent with Phase 1

**Node Subsampling Strategy:**
- Random uniform sampling (simplest, unbiased)
- The specific M nodes dropped vary per epoch (data augmentation effect)
- For GRED/hybrid: distance masks are recomputed for the subgraph

**Key Design Decision — Frozen Encoder:**
The encoder must remain frozen so that node embeddings in Phase 2 live in the same space as Phase 1. This is critical for Phase 3, where the generator must produce embeddings compatible with the Phase 1 representation space.

**Output:**
- `phase2_best.pt`: Best partial-graph model checkpoint
- Performance gap: Phase 1 AP − Phase 2 AP (expected to be positive)

**Expected Behavior:**
- AP should be lower than Phase 1 (less information available)
- The gap represents the information lost by removing M nodes
- This gap is what Phase 3's generator will attempt to recover

---

### Phase 3: Proxy Generator Training (Reconstruction)

**Objective:** Train a generator that takes N−M node embeddings as input and produces M proxy embeddings that approximate the missing nodes.

**Setup:**
- Load Phase 1 encoder (frozen) — used to produce target embeddings
- Load Phase 2 model (frozen) — used for downstream task loss
- Generator receives (N−M) node embeddings and must generate M proxy embeddings

**Training Signal (dual loss):**
1. **Reconstruction loss:** MSE between generated proxies and the actual embeddings of the held-out M nodes (from Phase 1 frozen encoder). This directly encourages in-distribution generation.
2. **Task loss:** Pass (N−M) + generated M through the Phase 2 frozen transformer and compute BCE on the classification task. This ensures proxies are not just reconstructing arbitrary features but are *useful* for the downstream task.

```
L_total = L_task + λ_recon * L_reconstruction
```

Where `λ_recon` is a hyperparameter (default: 1.0, annealed to 0.1 over training).

**Proxy-Target Alignment:**
Since the M held-out nodes are an unordered set, we use **Hungarian matching** (linear assignment) to align generated proxies with target nodes before computing reconstruction loss. This avoids the generator needing to learn a specific ordering.

**Generator Architecture:**
All existing generators (`ScoreBasedGenerator`, `PMAGenerator`, `GraphCoarseningGenerator`, `GNNPoolingGenerator`) are compatible — they take (B, N, d) node embeddings + mask and produce (B, M, d) proxy embeddings.

**Output:**
- `phase3_generator.pt`: Best generator checkpoint
- Validation: AP when using (N−M) + generated M through Phase 2 transformer

**Success Criterion:**
- Phase 3 AP (with generated proxies) > Phase 2 AP (without proxies)
- Ideally approaches Phase 1 AP (full graph)

---

### Phase 4: Augmented Evaluation

**Objective:** Use the trained generator to augment *complete* graphs with additional in-distribution nodes.

**Setup:**
- Load Phase 1 model (original full-graph transformer)
- Load Phase 3 generator
- For each graph:
  1. Encode all N nodes with Phase 1 encoder → (B, N, d)
  2. Generate M additional proxy nodes: generator(N nodes) → (B, M, d)
  3. Pass N+M nodes through Phase 1 transformer layers
  4. Classify using pooled representations

**Key Insight:**
During Phase 3, the generator learned to produce embeddings that look like real nodes. Now we give it the *full* graph (not a subset), so it generates M *additional* high-quality node-like tokens. Since these tokens are in-distribution, the transformer processes them naturally alongside the original N nodes.

**Expected Behavior:**
- AP should be ≥ Phase 1 baseline (more information, all in-distribution)
- The M generated proxies act as "virtual nodes" that capture complementary graph-level patterns

**Output:**
- Phase 4 evaluation AP on val/test sets
- Comparison table: Phase 1 vs Phase 2 vs Phase 3 vs Phase 4

---

### Phase 5: End-to-End Fine-tuning

**Objective:** Fine-tune the complete pipeline (transformer + generator) jointly for maximum performance.

**Setup:**
- Initialize from Phase 1 model + Phase 3 generator
- Two-phase fine-tuning (same as existing Stage 3):
  - **Phase A** (K epochs): Freeze transformer, fine-tune generator at reduced LR
  - **Phase B** (remaining epochs): Unfreeze transformer at lower LR, continue generator training
- Proxy dropout: randomly drop all proxies for a fraction of batches

**Loss:** Task loss only (BCE). The reconstruction loss from Phase 3 is dropped — the representation spaces will co-evolve during fine-tuning.

**Output:**
- `phase5_best.pt`: Final best model + generator checkpoint
- Final AP on val/test sets

---

## Architecture Details

### Node Subsampling (Phase 2)

```python
def subsample_nodes(dense_x, dense_mask, num_drop):
    """
    For each graph in the batch, randomly drop `num_drop` nodes.

    Args:
        dense_x: (B, max_N, d) node embeddings
        dense_mask: (B, max_N) boolean mask
        num_drop: number of nodes to drop (M)

    Returns:
        sub_x: (B, max_N, d) with dropped nodes zeroed
        sub_mask: (B, max_N) with dropped nodes masked out
        dropped_indices: (B, M) indices of dropped nodes per graph
        dropped_embeddings: (B, M, d) embeddings of dropped nodes
    """
```

### Hungarian Matching (Phase 3)

```python
from scipy.optimize import linear_sum_assignment

def hungarian_reconstruction_loss(generated, targets):
    """
    Compute MSE with optimal assignment between generated and target proxies.

    Args:
        generated: (B, M, d) generated proxy embeddings
        targets: (B, M, d) target node embeddings (held-out)

    Returns:
        loss: scalar reconstruction loss
    """
    # Per-graph cost matrix: (M, M) pairwise MSE
    # Solve linear assignment per graph
    # Average matched MSE
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_drop` | `num_proxies` (M) | Nodes to drop in Phase 2 |
| `p2_lr` | 1e-3 | Phase 2 learning rate |
| `p2_patience` | 50 | Phase 2 early stopping patience |
| `p3_lr` | 1e-3 | Phase 3 generator learning rate |
| `p3_recon_weight` | 1.0 | Initial reconstruction loss weight |
| `p3_recon_anneal_to` | 0.1 | Final reconstruction loss weight |
| `p3_recon_anneal_epochs` | 100 | Epochs over which to anneal |
| `p3_patience` | 50 | Phase 3 early stopping patience |
| `p5_phase_a_epochs` | 20 | Phase 5 frozen-transformer warmup |
| `p5_lr_gen` | 1e-4 | Phase 5 generator LR |
| `p5_lr_model` | 1e-5 | Phase 5 transformer LR |
| `p5_proxy_dropout` | 0.1 | Fraction of batches without proxies |

---

## Implementation Plan

### Files Modified
- **generators.py**: No changes needed — existing generators work as-is
- **models.py**: No changes needed — existing models work as-is
- **data.py**: No changes needed

### New Files
- **main_indist.py**: Complete IDPL pipeline orchestrator (Phases 1–5)
  - Reuses `build_model()`, `build_generator()`, `get_loaders()` from existing code
  - Adds node subsampling logic
  - Adds Hungarian matching reconstruction loss
  - Adds Phase 4 evaluation mode

### Reused Modules
- `models.py`: `GraphTransformer`, `GREDEncoder`, `GREDHybridTransformer`, `NodeEncoder`
- `generators.py`: All generator classes
- `data.py`: `get_loaders()`, distance mask infrastructure
- `metrics.py`: `compute_macro_ap()`
- `mmd.py`: Optionally for regularization

---

## Expected Results

| Phase | Description | Expected AP Range |
|-------|------------|-------------------|
| Phase 1 | Full transformer baseline | ~0.65–0.68 |
| Phase 2 | Partial (N−M) nodes | ~0.60–0.64 (gap: 3–5%) |
| Phase 3 | (N−M) + generated M | ~0.63–0.67 (recovering gap) |
| Phase 4 | N + generated M (augmented) | ~0.66–0.70 (above baseline) |
| Phase 5 | End-to-end fine-tuned | ~0.68–0.72 (best) |

The key hypothesis is that Phase 4 should exceed Phase 1 because the generator produces additional in-distribution tokens that capture complementary information.

---

## Usage

```bash
# Run all phases
python main_indist.py --phase all --backbone vanilla_gt --generator score_based

# Run individual phases
python main_indist.py --phase 1
python main_indist.py --phase 2 --model_path checkpoints_indist/phase1_best.pt
python main_indist.py --phase 3 --model_path checkpoints_indist/phase1_best.pt \
                                --phase2_model_path checkpoints_indist/phase2_best.pt
python main_indist.py --phase 4 --model_path checkpoints_indist/phase1_best.pt \
                                --generator_path checkpoints_indist/phase3_generator.pt
python main_indist.py --phase 5 --model_path checkpoints_indist/phase1_best.pt \
                                --generator_path checkpoints_indist/phase3_generator.pt
```