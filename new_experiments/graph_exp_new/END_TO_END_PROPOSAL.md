# Alternate method for graph transformer improvement with proxies with deterministic generator trained end-to-end for task

## 3. Method

### 3.0 Pipeline Overview
```
Single-stage end-to-end training:
  Node Features → Embedding Layer → Proxy Generator (compresses N nodes → M proxies)
                                  → concat [X; B] → Transformer Layers → Readout → Classification

All components trained jointly from scratch with task loss.
Optional: Pretrain transformer alone first, then attach generator (Stage 1 + Stage 2 below).
```

The architecture is a standard graph transformer augmented with a **proxy generator module** inserted between the embedding layer and the transformer layers. The generator compresses the N node embeddings into M proxy embeddings, which are concatenated with the original embeddings and processed by the transformer as N+M tokens.

---

### 3.1 Architecture

The full model has three components trained jointly:

#### 3.1.1 Embedding Layer

Standard node/edge feature embedding: AtomEncoder / BondEncoder / nn.Embedding → X ∈ ℝ^{N×d}

Same as the baseline graph transformer.

#### 3.1.2 Proxy Generator Module

**Note:** Keep this module a separate class in a separate file. The input is node embeddings X ∈ ℝ^{N×d}. The output is proxy embeddings B ∈ ℝ^{M×d}.

The generator compresses the full set of node embeddings into M proxy embeddings, where each proxy is a learned nonlinear combination of the graph's own node features.

**Step 1 — Value projection.** Project node embeddings into a value space (or skip this and use X directly, but a separate projection gives more capacity):

V = X · W_v, where W_v ∈ ℝ^{d×d}, so V ∈ ℝ^{N×d}

**Step 2 — Compute M score vectors.** A single shared MLP that outputs M scores per node. Each output column is a different learned scoring function over nodes:

S = MLP(X) ∈ ℝ^{N×M}

where the MLP is something like X·W₁ + b₁ → ReLU → ·W₂ + b₂, with W₁ ∈ ℝ^{d×h}, W₂ ∈ ℝ^{h×M}. So each node gets M scores, one per prototype.

**Step 3 — Softmax over nodes.** Normalize each prototype's scores across the node dimension so they form attention weights:

A = softmax(S, dim=0) ∈ ℝ^{N×M}

Each column of A sums to 1. Column m gives prototype m's soft selection over the N nodes. Crucially, this is entirely per-graph — the scores depend on X.

**Step 4 — Weighted aggregation.** Each prototype is a weighted sum of node values:

B₀ = Aᵀ · V, i.e. (M×N) · (N×d) → B₀ ∈ ℝ^{M×d}

**Step 5 — Self-attention among prototypes.** Let the M prototypes see each other so they can specialize and avoid redundancy:

B₁ = LayerNorm(B₀ + SelfAttn(Q=B₀, K=B₀, V=B₀)) ∈ ℝ^{M×d}

**Step 6 — FFN for nonlinearity:**

B₂ = LayerNorm(B₁ + FFN(B₁)) ∈ ℝ^{M×d}

where FFN is the usual Linear(d→4d) → GELU → Linear(4d→d).

Stack steps 5–6 for L layers (input param) for more refinement. B₂ ∈ ℝ^{M×d} is the final set of proxy embeddings.

**For batching** (graphs have different N): use dense batching with padding and a mask, same as the other setup. Mask out padding positions to −∞ before the softmax so padding nodes get zero weight.

Keep the architecture configurable (num layers, dropout percentage, num heads, hidden dim etc).

#### 3.1.3 Transformer Layers

Standard transformer stack processing N+M tokens:
```
[X; B] ∈ ℝ^{(N+M)×d} → Transformer Layers → global readout (mean pool over N original nodes) → classification head
```

Same architecture as the baseline graph transformer, but operating on the augmented token set. The readout pools over the original N nodes only (not the proxies), or over all N+M tokens (ablate both).

---

### 3.2 Training

#### 3.2.1 End-to-End Training (Primary)

Train the full model (embedding layer + proxy generator + transformer + classification head) jointly from scratch with the task loss:
```
L = L_task(readout(Transformer([X; Generator(X)])), y)
```

All parameters are updated with a single optimizer. No frozen components, no staged training.

---

### 3.3 Evaluation and Early Stopping

Every epoch:
1. For each graph in the validation set, run the full forward pass (generate proxies + transformer).
2. Compute validation AP.
3. Log: train task loss, train MMD loss (if used), **val AP**, **test AP**.
4. Save the checkpoint with the best val AP.

Early stop on val AP with patience 30 epochs.

**Diagnostic: Train loss vs val AP correlation.**

Plot training task loss against val AP across training.

---

### 3.4 Edge Prediction and GNN Integration

Instead of (or in addition to) inserting proxies into the transformer's attention, connect them to the original graph and use a GNN.

**Edge prediction options (input parameter to choose a mode):**

| Method | Description | Cost |
|---|---|---|
| Full connectivity | Connect each proxy to all N nodes | O(NM) edges — cheap for small M |
| Attention-weighted | Use softmax dot product of embeddings from generator outputs as edge weights (MxN dot product) | O(NM), soft, no extra parameters |
| Sigmoid dot-product | σ(Linear(b_i) · Linear(x_j)) for proxy i, node j | O(NM), learnable threshold |
| MLP classifier | MLP([b_i; x_j]) → {0,1} for each pair | O(NM) forward passes — feasible for small M |

**GNN integration:** Run a GNN on the augmented graph with proxy-to-node edges. This tests whether proxy quality transfers from the transformer setting to the GNN setting.

---

## 4. Inference Pipeline

At test time (for unseen graphs):
```
1. Compute node embeddings X via the embedding layer
2. Generate proxy embeddings B = Generator(X) (single forward pass)
3. Transformer processes [X; B]
4. Readout and classify
```

No per-graph optimization at inference. Single forward pass through the full model.

---

## 5. Experimental Plan

### 5.1 Ablations

Make all these parameters configurable via input args for ablations **later**.

- **M (number of proxies):** M ∈ {2, 4, 8, 16, 32}
- **Attention routing:** Full self-attention (N+M) vs. routed cross-attention (N→M→N) — both work comparably, but ablate formally
- **MMD regularization:** ablate λ ∈ {0, 0.005, 0.01, 0.05, 0.1}
- **End-to-end vs pretrain-then-attach:** Compare Section 3.2.1 vs 3.2.2
- **Generator depth:** L ∈ {0, 1, 2, 4} self-attention refinement layers
- **Readout scope:** Mean pool over N nodes only vs. all N+M tokens
- **Warmup:** Train without proxies for K epochs before activating generator, K ∈ {0, 5, 20, 50}

### 5.2 Metrics

Log these at appropriate intervals

- **Primary:** Average Precision (AP) on Peptides-func (multi-label classification) (every 1000 steps and after every epoch for validation and test sets)
- **Secondary:** Training/inference time, parameter count, memory usage (Every epoch)
- **Diagnostic:** Train loss vs downstream val AP correlation (Calculated considering every 1000 steps as new sample)

### 5.3 Additional Datasets (To be integrated later)

- Peptides-struct (LRGB), PascalVOC-SP (LRGB), PASCALVOC-SP, COCO-SP, PCQM-CONTACT, CIFAR10, MNIST, TreeNeighborsMatch (synthetic), ZINC

---

## 6. Implementation Notes

### 6.1 Peptides-func Specifics

- **Dataset size:** ~15,535 graphs (train/val/test split from LRGB).
- **Task:** 10-class multi-label binary classification.
- **Metric:** Average Precision (AP), macro-averaged.
- **Graph size:** Typically 100–300 nodes per graph.

### 6.2 Suggested Hyperparameter Starting Points (Modify as needed)
```yaml
# Full Model (End-to-End)
hidden_dim: 64
num_transformer_layers: 5
num_heads: 8
dropout: 0.3
weight_decay: 3e-4
batch_size: 256
lr: 1e-3
early_stopping_patience: 30
max_epochs: 500

# Proxy Generator
num_proxies_M: 32
generator_hidden_dim: 128        # hidden dim h for the scoring MLP
generator_refinement_layers: 2   # L self-attention layers among prototypes
generator_heads: 4
generator_dropout: 0.2
mmd_lambda: 0.01                 # 0 to disable
proxy_warmup_epochs: 0           # epochs to train without proxies first

# Pretrain-then-Attach (if used)
pretrain_max_epochs: 500
pretrain_early_stopping: 5
attach_lr_generator: 5e-4
attach_lr_transformer: 1e-5     # for optional unfreezing
```

### 6.3 Code Structure

Keep the structure simple, no sub directories and new files for small modules/functions. All model classes in one file (except the proxy generator module which is a separate file), data loading in one file and training loop for all phases in the main file along with input loading. No separate files or functions which set random seeds or setup logging. Minimal code focusing on functionality and accurate implementation.

For logging, use print statements with flush=True only. NO WANDB or MLFlow etc tools.

For config loading, keep an option to load from yaml files but main method should be command line args. Even if yaml config is passed, cli args should be overriding the input parameter.

---