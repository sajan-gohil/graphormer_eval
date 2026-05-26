# Baseline Experiments & Implementation Plan


## 1. Problem Statement

Graph Transformers and Message-Passing GNNs each have fundamental limitations on long-range dependency tasks:

**Graph Transformers** apply global self-attention over all nodes, capturing arbitrary pairwise interactions. However, full attention over N nodes introduces **global attention noise** — every node attends to every other node, including irrelevant ones — which degrades signal quality. They also discard graph topology, treating the input as a set rather than a structured graph.

**Message-Passing GNNs** respect graph topology but suffer from **oversquashing** (information bottlenecks at narrow graph cuts) and **oversmoothing** (node representations converging as depth increases). Both pathologies worsen on long-range tasks where information must traverse many hops.

Neither paradigm alone handles long-range graph dependencies well.

**Target benchmark:** Peptides-func (multi-label graph classification from LRGB/PyG), a dataset specifically designed to evaluate long-range interaction modeling in graphs.

---

## 2. Core Idea: Proxy Nodes as Information Hubs

### 2.1 Key Insight

Insert a small set of **M proxy nodes** (M ≪ N) into the graph. If well-positioned in embedding space, these proxies act as information hubs — any two nodes can communicate via a shared proxy, reducing the effective pairwise distance from O(diameter) to **O(2)**.

Unlike virtual nodes (a single global node connected to all), multiple proxies can **specialize**: different proxies can mediate different types of long-range interactions, reducing the information bottleneck that a single virtual node creates.

Unlike full global attention, routing through M proxies reduces attention complexity from O(N²) to O(NM) and focuses attention on learned "important" intermediate representations rather than all pairwise combinations.

### 2.2 Proxy Nodes as Hyperedges

The proxy nodes are conceptually equivalent to **hyperedges** in a hypergraph formulation. Each proxy node defines a hyperedge connecting a subset of (or all) original nodes. Attention is routed *through* these hyperedges:

1. **Node → Proxy cross-attention**: N nodes attend to M proxies (or vice versa), aggregating information into the proxy embeddings.
2. **Proxy → Node cross-attention**: The updated proxy embeddings broadcast information back to the original nodes.

This two-step routing is equivalent to message passing on a bipartite graph between nodes and hyperedges, with attention weights defining soft hyperedge membership.

### 2.3 The Challenge

We do not know a priori:
- What the proxy embeddings should be (they don't correspond to existing nodes)
- Which original nodes each proxy should connect to

Both must be **learned** and must **generalize to unseen graphs** at test time. This motivates using a conditional generative model — specifically, a conditional flow matching model — to produce proxy embeddings conditioned on the input graph.


## Purpose

Before building the flow matching pipeline, we need to answer two questions:

1. **Can proxy embeddings help at all?** (Validation gate — do optimized free embeddings improve a frozen transformer by ≥5 AP on train?)
2. **Is conditional generation necessary?** (Do simpler, non-generative proxy schemes already close the gap, making diffusion/flow matching overkill?)

This document specifies every experiment, what the coding agent should implement, and the decision logic at each stage.

---

## Experiment Roadmap

```
Phase 1: Train base transformer → establish baseline AP
Phase 2: Validation gate → optimize free embeddings, confirm ≥5 AP gain on train
Phase 3: Simple proxy baselines → test if non-generative methods suffice
Phase 4: Ablations on proxy optimization → understand what matters
Phase 5: Decision → proceed to flow matching or pivot
```

---

## Phase 1: Base Transformer Training

### Goal
Train a well-tuned graph transformer on Peptides-func and record the baseline AP. This model is reused in all subsequent experiments.

### What to Implement

For reference related to batching and encoding, refer `vanilla_gt.py`, specifically how `to_dense_batch`, `dense_mask`, `AtomEncoder` and `BondEncoder` are used, along with dataset splits and data loading.

**1.1 Data pipeline (`data/peptides_func.py`)**

- Load Peptides-func from `torch_geometric.datasets.LRGBDataset`.
- Use the official train/val/test split.
- Implement a `DataLoader` with batching via `torch_geometric.loader.DataLoader`.
- Compute and cache positional/structural encodings: Laplacian eigenvectors (k=8), random walk structural encoding (walk lengths up to 20).

**1.2 Model (`models/transformer.py`)**

- Implement GPS (General, Powerful, Scalable graph transformer) as the primary architecture. GPS interleaves local MPNN layers with global self-attention, making it a natural fit for proxy insertion (proxies participate in the global attention but not the local MPNN).
- Components needed:
  - `GPSLayer`: local MPNN (GIN or PNA) + global multi-head self-attention + FFN, with residual connections and LayerNorm.
  - `GPSModel`: embedding layer (atom/bond features → d-dim) + positional encoding injection + stack of GPSLayers + global readout (mean pool) + classification head (MLP → 10 outputs, sigmoid).
- Key hyperparameters to expose: `hidden_dim`, `num_layers`, `num_heads`, `dropout`, `attn_dropout`, `local_gnn_type` (GIN vs PNA).
- The embedding layer output (post-PE injection, pre-first-GPSLayer) must be extractable — this is the conditioning input for later stages. Add a hook or method `model.get_initial_embeddings(batch)` that returns the N×d tensor.

**1.3 Training loop (`training/pretrain.py`)**

- Optimizer: AdamW with weight decay 1e-5.
- LR scheduler: cosine annealing with warmup (10 epochs warmup).
- Early stopping on validation AP with patience 25 epochs, max 300 epochs.
- Logging (per epoch): train loss, train AP, val loss, val AP, learning rate. Log per-class AP on validation set every 10 epochs.
- Save best model checkpoint (by val AP) and the corresponding epoch.
- At the end of training, evaluate on test set and log test AP.

**1.4 Metrics (`evaluation/metrics.py`)**

- Implement AP computation compatible with Peptides-func: use `sklearn.metrics.average_precision_score` with `average='macro'` over the 10 classes (matching LRGB evaluation protocol).
- Also implement per-class AP for diagnostic purposes.
- Implement a function that, given predictions and labels, returns the set of classes correctly predicted per sample (using threshold 0.5), used later for the "at least one more class" filtering criterion.

### Expected Output

A table like:

| Metric | Value |
|---|---|
| Train AP | ~0.72–0.78 |
| Val AP | ~0.65–0.68 |
| Test AP | ~0.64–0.67 |
| Best epoch | ~80–150 |
| Training time | ~2–4 hours |

These numbers are approximate. The key is a well-trained, non-overfitting baseline.

---

## Phase 2: Validation Gate

### Goal
Confirm that optimized proxy embeddings can improve the frozen transformer by ≥5 AP points on training data.

### What to Implement

**2.1 Proxy optimizer (`models/proxy_optimizer.py`)**

- Class `ProxyOptimizer`:
  - Takes a frozen `GPSModel`, a graph batch, and config (M, lr, num_iterations, mmd_lambda, gradient_clip).
  - Initializes `B = nn.Parameter(torch.randn(M, d) * sigma)` where `sigma` is computed from the empirical std of the node embeddings in the batch.
  - Forward pass: extract initial embeddings X from the frozen model, concatenate B as additional "nodes" (handling batch indexing correctly — each graph in the batch gets its own copy of B), run through the frozen GPS layers, apply readout and classification head.
  - **Batch indexing detail:** In PyG, `batch.batch` is a vector assigning each node to its graph. When adding M proxies per graph, create a proxy batch assignment vector and concatenate. The attention mask must be updated so proxies only attend within their graph. This is the trickiest implementation detail - refer how batches are made in `vanilla_gt.py` and how it is passed to attention and combined back for other parts of the code. Any graph level additions to the nodes, like adding proxies will require appropriate masks for padding nodes between the original and proxies, any updates to batch.batch, etc.
  - Loss: `L_task + lambda * MMD²(B, X)`.
  - Optimizer: Adam on B only, with gradient clipping.
  - Returns: optimized B, final loss, AP improvement.

- MMD implementation (`utils/mmd.py`):
  - Gaussian kernel MMD²: `MMD²(P, Q) = E[k(p,p')] + E[k(q,q')] - 2E[k(p,q)]` with `k(x,y) = exp(-||x-y||² / (2σ²))`.
  - Use median heuristic for kernel bandwidth σ.

**2.2 Validation script (`training/validate_premise.py`)**

- Load the best pretrained model from Phase 1. Freeze all parameters.
- Select a representative subset of training graphs (500–1000, stratified by class distribution).
- For each graph:
  - Run 5 random restarts of `ProxyOptimizer` with M=8, 500 iterations each.
  - Keep the B with lowest final loss.
  - Record: baseline loss (no proxy), optimized loss (with proxy), baseline AP, optimized AP, classes correctly predicted before vs after.
- Aggregate results:
  - Mean AP improvement across the subset.
  - Distribution of per-graph AP improvements (histogram).
  - Fraction of graphs that improved by ≥0.
  - Per-class AP before vs after.
- Repeat with M ∈ {2, 4, 8, 16} to understand sensitivity.

**2.3 Diagnostic logging**

For a small subset (50 graphs), additionally log:
- Optimization trajectory: loss and AP at every 50th iteration.
- Final B embedding norms and pairwise cosine similarities.
- MMD²(B, X) at initialization vs at convergence.
- Attention weight heatmap: which original nodes attend most to which proxies (extract from the last GPS layer's global attention).

---

## Phase 3: Simple Proxy Baselines

### Goal
Establish how much of the proxy benefit can be captured by non-generative methods. This determines the "bar" the flow matching model must clear.

### What to Implement

**3.1 Virtual Node Baseline (`models/virtual_node.py`)**

- Implement a single virtual node connected to all nodes in the graph.
- The virtual node participates in every GPS layer's global attention.
- Two variants:
  - **VN-Fixed:** Virtual node embedding is a fixed learnable parameter (shared across all graphs). Initialized from N(0, σ).
  - **VN-Aggregated:** Virtual node embedding is initialized as the mean of node embeddings at each layer, then updated via attention. (This is the standard virtual node approach.)
- Train the full model (transformer + virtual node) from scratch with the same hyperparameters as Phase 1.
- Report train/val/test AP.

**3.2 K Fixed Virtual Nodes Baseline (`models/k_virtual_nodes.py`)**

- Extend the virtual node to M fixed learnable embeddings, shared across all graphs.
- These are `nn.Parameter(torch.randn(M, d))`, initialized once and optimized during training as part of the model parameters.
- Each graph receives the same M proxy embeddings, regardless of its content.
- They participate in global self-attention at every GPS layer.
- This is the critical baseline — it tests whether graph-conditional generation adds value over fixed tokens.
- Train from scratch, try M ∈ {4, 8, 16}.
- Report train/val/test AP.

**3.3 Mean Pooling Proxies (`models/mean_proxy.py`)**

- Generate M proxy embeddings per graph by simple aggregation of node embeddings, without any learning:
  - **Global mean:** All M proxies = mean(X). (Degenerate, but establishes a floor.)
  - **K-means centroids:** Run k-means on X with k=M; use centroids as proxies.
  - **Random node sampling:** Randomly select M nodes from X as proxies.
  - **Degree-weighted sampling:** Sample M nodes with probability proportional to node degree.
- Insert these into the frozen pretrained transformer (from Phase 1) and evaluate on train/val/test.
- No additional training — this tests whether naive content-aware proxy placement helps.

**3.4 Random Proxy Embeddings (`models/random_proxy.py`)**

- Generate M random vectors from N(μ_X, σ_X) (matching the node embedding distribution).
- Insert into the frozen pretrained transformer and evaluate.
- This ablates whether content matters at all, or if any additional tokens help via regularization.
- Run 10 random seeds and report mean ± std of AP.

### Results Table to Produce

| Method | M | Train AP | Val AP | Test AP | Params Added | Notes |
|---|---|---|---|---|---|---|
| GPS (no proxy) | — | — | — | — | 0 | Phase 1 baseline |
| VN-Fixed | 1 | — | — | — | d | Standard approach |
| VN-Aggregated | 1 | — | — | — | d | Standard approach |
| K-Fixed-VN | 4 | — | — | — | 4d | |
| K-Fixed-VN | 8 | — | — | — | 8d | Critical comparison |
| K-Fixed-VN | 16 | — | — | — | 16d | |
| Set-Transformer-IP | 4 | — | — | — | ~4d + attn | |
| Set-Transformer-IP | 8 | — | — | — | ~8d + attn | |
| Mean-proxy (global) | 8 | — | — | — | 0 | Frozen model |
| Mean-proxy (k-means) | 8 | — | — | — | 0 | Frozen model |
| Mean-proxy (random nodes) | 8 | — | — | — | 0 | Frozen model |
| Random embeddings | 8 | — | — | — | 0 | Frozen model, 10 seeds |
| Optimized proxies (Stage 0) | 8 | — | N/A | N/A | 0 | Upper bound, per-graph opt |


---

## Phase 4: Ablations on Proxy Optimization

### Goal
Understand which design choices in proxy optimization matter most, informing the flow matching target generation (Stage 2 of the main pipeline).

### What to Implement

**4.1 Insertion layer ablation**

- Instead of inserting proxies at the input (before layer 0), try inserting at:
  - After layer 1 (proxies skip the first layer)
  - After layer 2
  - At all layers simultaneously (proxies are added fresh at each layer, optimized jointly)
- Implement as a config option in `ProxyOptimizer`: `insertion_point ∈ {0, 1, 2, "all"}`.

**4.2 Attention routing ablation**

- Compare three routing modes:
  - **Full:** Standard self-attention on (N+M) — proxies and nodes see each other bidirectionally.
  - **Routed:** Attention is node→proxy then proxy→node only. Nodes don't directly attend to each other. Implementation: separate matrices for node and proxy batch embeddings with appropriate batch matmuls.
  - **Hybrid:** Full node-node attention + cross-attention with proxies. Two separate attention operations per layer.
- Implement as attention modifications in the GPS layer.

**4.3 MMD sensitivity**

- Run proxy optimization with λ_MMD ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}.
- For each λ, track and plot: final task loss, final AP, final MMD²(B, X), B norm statistics.
- Plot the Pareto frontier of task performance vs distribution alignment.


### Results to Produce

- Table of AP for each ablation dimension.
- Recommendation for the best configuration to use in Stage 2 target generation.
- Visualization: t-SNE or UMAP of optimized proxy embeddings alongside original node embeddings, colored by graph identity, for 20–30 graphs. This shows whether proxies cluster by graph or whether they're similar across graphs (which would favor fixed tokens).

---

## Phase 5: Decision Point

### Synthesize All Results

Produce a summary table:

| Experiment | Train AP | Val AP | Test AP | Key Insight |
|---|---|---|---|---|
| GPS baseline | — | — | — | Starting point |
| Optimized proxies (best config) | — | — | — | Upper bound |
| K-Fixed-VN (best M) | — | — | — | Fixed token ceiling |
| Set-Transformer-IP (best M) | — | — | — | Cross-attn variant |
| Virtual Node | — | — | — | Standard approach |
| Random proxies | — | — | — | Noise floor |

### Go/No-Go Criteria for Flow Matching

**GO** if all of the following hold:
- Optimized proxies improve train AP by ≥5 points over GPS baseline.
- Optimized proxies outperform K-Fixed-VN by ≥3 points on train AP, indicating that per-graph conditioning matters.
- The embedding visualization (t-SNE) shows meaningful variation in proxy embeddings across graphs, confirming that one-size-fits-all tokens are suboptimal.

**PIVOT** if:
- Optimized proxies help, but K-Fixed-VN captures most of the gain. In this case, write a paper about learned virtual nodes with analysis of why they work, rather than building a generative model.
- Optimized proxies help significantly, but proxy embeddings are very similar across graphs. In this case, a simple encoder (MLP on graph-level features → proxy embeddings) may suffice instead of flow matching.

**STOP** if:
- Optimized proxies don't meaningfully improve the frozen transformer (< 2 AP points).

---

## Implementation Checklist for Coding Agent

### Priority 1 — Must Have (Phases 1–2)

```
□ data/peptides_func.py
    □ Load from PyG LRGBDataset
    □ Compute & cache Laplacian PEs, RWSE
    □ DataLoader with PyG batching
    □ Dataset statistics logging

□ models/transformer.py
    □ GPSLayer (local GIN/PNA + global self-attn + FFN)
    □ GPSModel (embedding + PE + GPS stack + readout + classifier)
    □ model.get_initial_embeddings(batch) method
    □ Configurable: hidden_dim, num_layers, num_heads, dropout

□ training/pretrain.py
    □ AdamW + cosine LR with warmup
    □ Early stopping on val AP, patience 25
    □ Per-epoch logging: train/val loss and AP
    □ Save best checkpoint
    □ Final test evaluation

□ evaluation/metrics.py
    □ Macro-averaged AP (sklearn)
    □ Per-class AP
    □ Correct-class set computation (threshold 0.5)

□ utils/mmd.py
    □ Gaussian kernel MMD² with median heuristic

□ models/proxy_optimizer.py
    □ ProxyOptimizer class
    □ Proper batch indexing for proxy insertion
    □ MMD regularization
    □ Multiple restarts
    □ Diagnostic logging (trajectory, norms, attention maps)

□ training/validate_premise.py
    □ Run ProxyOptimizer on stratified subset
    □ Aggregate results, produce summary table
    □ M sensitivity (M ∈ {2, 4, 8, 16})
```

### Priority 2 — Baselines (Phase 3)

```
□ models/virtual_node.py
    □ VN-Fixed variant
    □ VN-Aggregated variant
    □ Integration with GPSModel

□ models/k_virtual_nodes.py
    □ M fixed learnable embeddings
    □ Proper batch insertion
    □ M ∈ {4, 8, 16}

□ models/mean_proxy.py
    □ Global mean variant
    □ K-means centroids variant
    □ Random node sampling variant
    □ Degree-weighted sampling variant
    □ Insert into frozen model & evaluate

□ models/random_proxy.py
    □ Sample from N(μ_X, σ_X)
    □ 10-seed evaluation
```

### Priority 3 — Ablations (Phase 4)

```
□ Insertion layer ablation
    □ Modify GPSModel to support proxy insertion at arbitrary layers
    □ "all layers" variant

□ Attention routing ablation
    □ Full / Routed / Hybrid attention masks
    □ Cross-attention implementation for routed variant

□ MMD sensitivity sweep
    □ λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}

□ Visualization
    □ t-SNE/UMAP of proxy + node embeddings
    □ Proxy diversity analysis across graphs
```

### Priority 4 — Infrastructure

```
□ configs/
    □ JSON or YAML config for each experiment
    □ Sweep configs for hyperparameter searches

□ Experiment tracking
    □ NO API based tools, only log files.
    □ Table/figure generation from metrics

□ scripts/
    □ run_phase1.sh, run_phase2.sh, run_phase3.sh, run_phase4.sh
    □ Each script runs the corresponding phase end-to-end

□ tests/
    □ Test batch indexing for proxy insertion (critical correctness check)
    □ Test MMD computation
    □ Test AP metric matches LRGB reference
    □ Test that frozen model parameters don't change during proxy optimization
```

---

## Notes for the Coding Agent

**correctness issue:** Proxy insertion with proper batch indexing in PyG. In PyG's batching, all graphs in a batch are concatenated into a single large graph, with `batch.batch` tracking which node belongs to which graph. When inserting M proxies per graph, you must: create M new node entries per graph in the batch, assign them correct `batch.batch` values, ensure the attention mask prevents proxies from one graph attending to nodes in another graph, handle the readout (mean pool) correctly — proxies should probably be excluded from readout unless you explicitly want them included. Get this wrong and all results will be garbage. Write a unit test that verifies: a batch of 3 graphs with N1, N2, N3 nodes and M=2 proxies each produces an augmented batch with (N1+2), (N2+2), (N3+2) nodes, and that attention is correctly scoped. Using `to_dense_batch` will add padding nodes, to smaller graphs, so appropriate masks should be there which are used when computing attention with or without proxies.

**issue:** Making sure the pretrained model's parameters are truly frozen during proxy optimization. Use `model.eval()` and `param.requires_grad_(False)` for all model parameters. Verify with a unit test: save a hash of all parameter tensors (could just be first few values, or values at random indices) before and after proxy optimization; they must match exactly. This should still allow gradients to pass through them to the learnable proxy node and proxy nodes should get updated.

**Reproducibility:** Set random seeds at the top of every script (`torch.manual_seed`, `np.random.seed`, `random.seed`). Use deterministic algorithms where possible (`torch.use_deterministic_algorithms(True)`).