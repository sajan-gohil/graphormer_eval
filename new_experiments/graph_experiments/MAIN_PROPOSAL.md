# Diffusion-Generated Proxy Nodes for Graph Transformers

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

---

## 3. Method

### 3.1 Pipeline Overview

```
Stage 0: VALIDATION — Verify proxy embeddings can help (≥5 AP improvement on train)
Stage 1: Pretrain Graph Transformer (frozen after this stage)
Stage 2: Optimize proxy embeddings per graph (produce flow matching targets)
Stage 3: Train conditional flow matching model to generate proxy embeddings
Stage 4: (Optional) End-to-end fine-tuning of full pipeline
Stage 5: (Optional) Edge prediction / GNN integration
```

---

## 3.2 Stage 0 — Validation Gate (MUST PASS BEFORE PROCEEDING)

**Purpose:** Before building any generative model, we must verify the foundational premise — that inserting optimized proxy embeddings into a frozen transformer can meaningfully improve performance. If free embeddings optimized directly against the task loss cannot help, no generative model will either.

**Pass criterion:** Optimized proxy embeddings must improve **training set AP by ≥5 points** over the proxy-free baseline.

**Procedure:**

1. Train a graph transformer on Peptides-func with early stopping. Record train AP and validation AP. This becomes the baseline.
2. Freeze all transformer parameters.
3. For a representative subset of training graphs (e.g., 500–1000 graphs, stratified by class distribution):
   a. Initialize M learnable embedding vectors B ∈ ℝ^{M×d}, sampled from N(0, σ) where σ matches the empirical standard deviation of the node embeddings at the insertion layer.
   b. Pass (N + M) embeddings through the frozen transformer using full self-attention.
   c. Compute the task loss (binary cross-entropy).
   d. Update only B via Adam (lr=1e-2, gradient clipping at 1.0).
   e. Run for 500 iterations or until loss converges (change < 1e-6 for 20 consecutive steps).
   f. Repeat with 5 random initializations; keep the B that achieves the lowest loss.
4. Compute AP over the subset using the optimized proxies.
5. Compare to baseline AP on the same subset (without proxies).

**Decision matrix:**

| Train AP Improvement | Action |
|---|---|
| ≥ 5 AP points | **PASS.** Proceed to Stage 1. The premise holds — proxy embeddings can materially help. |
| 2–5 AP points | **Investigate.** Try increasing M, changing the insertion layer, or using cross-attention routing. If no configuration reaches ≥5, reconsider the approach. |
| < 2 AP points | **FAIL.** The premise does not hold for this architecture/dataset. Proxy embeddings do not provide enough signal. Abandon or fundamentally rethink. |

**Diagnostic checks during validation:**

- **Embedding drift:** Track the L2 norm and cosine similarity of B relative to X at each optimization step. If B drifts far from the node embedding distribution, the transformer's attention mechanism may not be handling the proxies well. Add MMD regularization and re-run.
- **Attention pattern analysis:** Visualize which original nodes attend most to which proxies. If all proxies receive uniform attention, they are not specializing — try larger M or different initialization.
- **Per-class improvement:** On Peptides-func (10 classes), check which classes improve. If only 1–2 easy classes improve, the proxies may not be capturing diverse long-range interactions.
- **M sensitivity:** Run the validation with M ∈ {2, 4, 8, 16}. If M=2 already saturates the improvement, fewer proxies suffice and the diffusion model's task is simpler. If improvement scales with M up to 16+, there's more room for the generative model to add value.

**Why ≥5 AP points on training data?** The diffusion model will inevitably produce imperfect approximations of the optimal proxies, so we need headroom. If optimal proxies give +5 on train, the generative approximation might give +2–3 on train and +1–2 on validation/test, which is still a publishable result. If optimal proxies only give +2 on train, the generated proxies may show no improvement after approximation error.

---

### 3.3 Stage 1 — Pretrain Graph Transformer

Train a standard graph transformer on Peptides-func. This transformer serves two purposes:
- It provides the **embedding space** in which proxy nodes will live.
- It provides the **frozen evaluation function** used to optimize proxy targets in Stage 2.

**Key considerations:**
- Use early stopping to prevent overfitting (the dataset overfits easily in 100–200 epochs). Monitor validation AP and stop when it plateaus or degrades. Use patience of ~20–30 epochs.
- Record the input embeddings to the attention layers (post positional encoding, post initial projection) — these are the conditioning inputs for the flow matching model.
- The architecture choice (e.g., GPS, GraphGPS, SAN, Graphormer) determines where and how proxy nodes will be inserted. GPS-style architectures that combine local MPNN + global attention are natural candidates.

Note: If Stage 0 was run with a quick-trained transformer, Stage 1 may involve retraining more carefully or simply reusing the Stage 0 model if it was already well-tuned.

### 3.4 Stage 2 — Optimize Proxy Embeddings (Target Generation)

**Goal:** For each training graph, find M proxy embeddings that, when inserted into the frozen pretrained transformer, improve the task loss. These optimized embeddings become the **regression targets** for the flow matching model.

**Procedure:**

1. Freeze all pretrained transformer parameters.
2. Initialize M learnable embedding vectors B ∈ ℝ^{M×d} (per graph), e.g., from N(0, σ) where σ matches the empirical standard deviation of the existing node embeddings.
3. Pass the augmented set of (N + M) embeddings through the frozen transformer.
4. Compute the task loss (binary cross-entropy for multi-label classification on Peptides-func).
5. Backpropagate gradients only through B; update B via gradient descent.
6. Repeat for T iterations or until convergence.

**Attention routing options to explore:**

| Variant | Description | Complexity |
|---|---|---|
| Full self-attention | Wq(N+M) @ Wk(N+M)ᵀ — treat proxies as additional nodes | O((N+M)²) |
| Routed cross-attention | (Wq(N) @ Wk(B)ᵀ) then (Wq'(B) @ Wk'(N)ᵀ) — attention flows through proxies only | O(NM) |
| Hybrid | Self-attention among original N nodes + cross-attention with M proxies | O(N² + NM) |

For compatibility with the frozen pretrained network, full self-attention on (N+M) is the most straightforward since it reuses existing attention weights without modification.

**Distribution constraint (MMD):**

```
L_total = L_task + λ · MMD²(B, X)
```

where MMD² uses a Gaussian kernel. This prevents proxy embeddings from drifting to arbitrary regions of embedding space that the transformer has never seen. Use λ as a tunable hyperparameter (start with 0.01–0.1). Optimal proxies may legitimately sit at cluster centroids or interpolation points that aren't strictly in-distribution, so don't over-constrain.

**Filtering targets:**

Only retain optimized proxy embeddings as flow matching targets when they demonstrably improve performance:
- The per-sample task loss decreases compared to the proxy-free baseline, **or**
- At least one additional class is correctly predicted (relevant for multi-label Peptides-func).

**Caveat:** Aggressive filtering can introduce bias. Consider also using soft weighting (weight each sample's flow matching loss by the improvement magnitude) rather than hard filtering.

**Using a subset of M proxies as targets:**

Optimize M=16 proxies per graph but use only the top-k (e.g., k=8) as flow matching targets. Identify the top-k by leave-one-out ablation: remove each proxy and measure loss increase. The proxies whose removal hurts most are the most informative targets. This gives the optimizer more degrees of freedom while producing cleaner targets for the generative model.

### 3.5 Stage 3 — Train Conditional Flow Matching Model

**Goal:** Train a generative model that, given the node embeddings X of any graph, produces M proxy embeddings B that improve the transformer's predictions on that graph.

#### 3.5.1 Why Flow Matching

Flow matching is the preferred generative framework for this pipeline for several interconnected reasons:

- **One-step generation at inference:** Flow matching learns a vector field that transports a simple prior z ~ N(0,I) to the target distribution. At inference, a single ODE step (or a few Euler steps) produces high-quality samples. This is critical for Stage 4 end-to-end finetuning where gradients must flow through the generation process.
- **Simple training objective:** The loss is a direct regression on the conditional vector field — no noise schedule tuning, no variance weighting, no EMA of target networks.
- **Clean backpropagation:** A single forward pass through the model produces the output. Gradients flow through one function evaluation, avoiding the memory explosion of multi-step DDPM denoising.
- **No mode coverage issues of VAEs:** Flow matching does not suffer from the posterior collapse or mode averaging that plagues VAEs, making it better suited for generating diverse proxy embeddings.

#### 3.5.2 Flow Matching Formulation

**Conditional Flow Matching (CFM)** defines a time-dependent vector field u_t that transports samples from a prior distribution p_0 = N(0, I) to the target distribution p_1 (the distribution of optimal proxy embeddings).

For a target sample x_1 (an optimized proxy embedding) and noise sample x_0 ~ N(0, I):

**Interpolation path (optimal transport):**

```
x_t = (1 - t) · x_0 + t · x_1,    t ∈ [0, 1]
```

**Conditional vector field:**

```
u_t(x_t | x_1) = x_1 - x_0
```

**Training objective:**

```
L_FM = E_{t~U(0,1), x_0~N(0,I), x_1~p_data} || v_θ(x_t, t | X) - u_t(x_t | x_1) ||²
```

where v_θ is the learned vector field (our Transformer denoiser), conditioned on the graph's node embeddings X.

**Sampling (inference):**

```
z_0 ~ N(0, I) ∈ ℝ^{M×d}
B = z_0 + v_θ(z_0, t=0 | X)          # One-step (Euler, t: 0→1)
```

Or for higher quality, use a few Euler steps:

```
z_0 ~ N(0, I)
for t in [0, Δt, 2Δt, ..., 1-Δt]:
    z_{t+Δt} = z_t + Δt · v_θ(z_t, t | X)
B = z_1
```

In practice, 1–4 Euler steps suffice for low-dimensional continuous data like embeddings.

#### 3.5.3 Architecture — Transformer Denoiser with Cross-Attention

```
Input:  x_t ∈ ℝ^{M×d}  (interpolated proxy tokens at time t)
Cond:   X ∈ ℝ^{N×d}     (original node embeddings, frozen)
Time:   t ∈ [0, 1]       (flow time, embedded via sinusoidal or MLP encoding)

Time embedding: t_emb = MLP(sinusoidal(t)) ∈ ℝ^d

For each denoiser layer:
    x_t ← x_t + t_emb                        # AdaLN or additive time conditioning
    x_t ← SelfAttention(x_t)                  # proxy tokens attend to each other
    x_t ← CrossAttention(Q=x_t, K=X, V=X)    # proxy tokens attend to node embeddings
    x_t ← FFN(x_t)

Output: v_θ(x_t, t | X) ∈ ℝ^{M×d}            # predicted vector field
```

The cross-attention mechanism allows each proxy token to selectively gather information from the relevant nodes in the input graph, enabling graph-conditioned generation.

**Time conditioning options:**
- **Additive:** Add t_emb to the input at each layer (simplest).
- **Adaptive LayerNorm (AdaLN):** Predict scale and shift of LayerNorm from t_emb (more expressive, used in DiT).
- **Recommendation:** Start with additive; switch to AdaLN if underfitting.

#### 3.5.4 Conditioning on Graph Structure

Beyond node embeddings X, consider conditioning on:
- **Graph-level features**: A global graph embedding (e.g., mean/max pool of X, or a readout from the pretrained transformer). Can be concatenated to t_emb.
- **Structural encodings**: Laplacian eigenvectors, random walk positional encodings, or degree features — already computed for the pretrained transformer. These can be appended to each node's embedding in the cross-attention keys/values.
- **Edge information**: Sparse attention or adjacency-aware cross-attention in the denoiser (e.g., using graph attention bias in the cross-attention layer).

Start with node embeddings only for simplicity; add structural conditioning if the generated proxies don't capture topology well enough.

#### 3.5.5 Training Details

- **Input:** Node embeddings X extracted from the pretrained transformer's input layer (the same embeddings the proxy targets were optimized against in Stage 2).
- **Targets:** Optimized proxy embeddings B* from Stage 2.
- **Loss:** Flow matching loss (MSE between predicted vector field and ground truth conditional vector field).
- **Time sampling:** t ~ U(0, 1). Optionally use logit-normal sampling (concentrating more samples near t=0 and t=1) if training is unstable.
- **Multiple targets per graph:** If Stage 2 produced K valid targets per graph (from K random restarts), sample uniformly from them during training. This effectively multiplies the dataset size and improves diversity.

#### 3.5.6 Normalization Protocol

The distribution of node embeddings fed to the transformer must remain consistent. If the flow matching model produces embeddings with different statistics than what the pretrained transformer expects, attention and layer norms will behave unexpectedly.

**Recommended approach:**
1. Before Stage 2, compute the per-dimension mean μ and std σ of all node embeddings across the training set.
2. Standardize: work in the space x̃ = (x - μ) / σ for both the proxy targets and the flow matching model.
3. After generation, denormalize: B = σ · B̃ + μ.
4. Apply the same LayerNorm (with frozen pretrained statistics) to generated proxies before insertion.
5. Monitor embedding norms and cosine similarities throughout training.

### 3.6 Stage 4 — End-to-End Fine-Tuning (Optional)

Fine-tune the full pipeline jointly:

```
Task Loss → Transformer → Flow Matching Generator → Embedding Layer
```

**Gradient flow with flow matching:** The generation step is a single forward pass through v_θ:

```
B = z_0 + v_θ(z_0, t=0 | X)
```

This is fully differentiable. Gradients from the task loss flow through:
1. The transformer (producing ∂L/∂B)
2. Through the addition, into v_θ (producing ∂L/∂θ_denoiser)
3. Through the cross-attention in v_θ, into X (producing ∂L/∂X)
4. Into the embedding layer (producing ∂L/∂θ_embed)

No multi-step unrolling, no adjoint methods, no gradient checkpointing tricks. This is the primary advantage of flow matching over DDPM for end-to-end training.

If using K Euler steps instead of one, backpropagation requires K forward passes through v_θ stored in memory. For K ≤ 4 with a small denoiser, this is manageable.

**What gets updated during fine-tuning:**
- Denoiser parameters (primary, normal learning rate)
- Transformer parameters (small learning rate ~1/10th, to avoid catastrophic forgetting)
- Embedding layer (small learning rate)

**Risks:** The transformer may adapt to rely on artifacts of the denoiser rather than genuine structural information. Monitor validation AP carefully and use aggressive early stopping.

### 3.7 Stage 5 — Edge Prediction and GNN Integration (Optional)

Instead of (or in addition to) inserting proxies into the transformer's attention, connect them to the original graph and use a GNN.

**Edge prediction options:**

| Method | Description | Cost |
|---|---|---|
| Full connectivity | Connect each proxy to all N nodes | O(NM) edges — cheap for small M |
| Attention-weighted | Use final cross-attention weights from flow matching as edge weights | O(NM), soft, no extra parameters |
| Sigmoid dot-product | σ(b_i · x_j) for proxy i, node j | O(NM), learnable threshold |
| MLP classifier | MLP([b_i; x_j]) → {0,1} for each pair | O(NM) forward passes — feasible for small M |

**Recommendation:** Start with attention-weighted soft connectivity (essentially free, extracted from the flow matching cross-attention). If that works, try the sigmoid dot-product for hard/sparse edges. The MLP approach is feasible given small M (e.g., M=10, N=150 for Peptides → 1500 pairs per graph).

**GNN integration:** With proxy-to-node edges established, run a GNN (e.g., GIN, GAT, PNA) on the augmented graph. The proxy nodes effectively create "shortcut" edges that reduce effective graph diameter. This tests whether the quality of learned proxies transfers from the transformer setting to the GNN setting.

---

## 4. Inference Pipeline

At test time (for unseen graphs):

```
1. Compute node embeddings X via the embedding layer
2. Sample noise z_0 ~ N(0, I) ∈ ℝ^{M×d}
3. Generate proxy embeddings: B = z_0 + v_θ(z_0, t=0 | X)   [one Euler step]
4. Denormalize if needed: B = σ · B + μ
5. Insert B into the transformer alongside X
6. Compute predictions
```

No per-graph optimization is needed at inference — the flow matching model generalizes from the training distribution.

**Multiple samples:** Since the generative model is stochastic (different z_0 samples), we can generate multiple sets of proxy embeddings and ensemble predictions. This may improve robustness at moderate compute cost.

---

## 5. Relation to Existing Work

| Method | Relation to This Work |
|---|---|
| **Virtual Nodes** (Gilmer et al., 2017) | Single global node connected to all. Our method uses M specialized proxies with learned connectivity — strictly more expressive. |
| **Set Transformer / Perceiver** (Lee et al., 2019; Jaegle et al., 2021) | Inducing points for attention compression. Similar O(NM) complexity, but inducing points are learned as fixed parameters, not generated per-graph. Our flow matching model conditions on each graph. |
| **Graph Coarsening / Pooling** (Ying et al., 2018; Bianchi et al., 2020) | Reduces graph to fewer nodes via clustering. Proxies are generated independently rather than by pooling existing nodes, and augment rather than replace the original graph. |
| **Expander Graphs for GNNs** (Deac et al., 2022) | Add random regular expander edges to improve connectivity. Our proxies achieve similar connectivity improvement but with learned, content-aware placement rather than random structure. |
| **Graph Transformers** (Ying et al., 2021; Kreuzer et al., 2021; Rampášek et al., 2022) | Our base model. We augment rather than replace global attention. |
| **Diffusion/Flow for Graph Generation** (Jo et al., 2022; Vignac et al., 2023) | Generate entire graphs. We generate only auxiliary proxy nodes conditioned on an existing graph — a much more constrained and targeted generative task. |
| **Flow Matching** (Lipman et al., 2023; Liu et al., 2023) | We adopt the OT-CFM framework as our generative backbone for proxy embedding generation. |

### Novelty Claim

The core novelty is using a **conditional flow matching model to produce graph-specific auxiliary nodes** that improve a downstream predictor. This differs from: fixed learnable tokens (inducing points, CLS tokens) which are the same for all inputs; graph coarsening which reduces rather than augments; virtual nodes which use a single unspecialized node; random structural augmentation (expander edges) which is not content-aware.

---

## 6. Experimental Plan

### 6.1 Baselines

(See companion document: "Baseline Experiments & Implementation Plan" for full details on establishing baselines before pursuing the flow matching pipeline.)

- **Pretrained Graph Transformer** (no proxies) — the Stage 1 model
- **Virtual Node** — single learnable global node connected to all
- **K Virtual Nodes** — M fixed learnable embeddings (same for all graphs)
- **Set Transformer Inducing Points** — M inducing points with cross-attention
- **Mean/Subgraph Pooling Proxies** — simple non-learned proxy generation
- **Random Proxy Embeddings** — M random vectors from matching distribution

### 6.2 Ablations

- **M (number of proxies):** Try M ∈ {2, 4, 8, 16, 32}
- **Attention routing:** Full self-attention (N+M) vs. routed cross-attention vs. hybrid
- **MMD regularization:** With and without, varying λ
- **Target filtering:** Hard filtering vs. soft weighting vs. no filtering
- **Structural conditioning:** Node embeddings only vs. + graph-level features vs. + structural encodings
- **End-to-end finetuning:** With and without Stage 4
- **Euler steps at inference:** 1 vs. 2 vs. 4 steps
- **Ensemble sampling:** 1 vs. 5 vs. 10 proxy samples at inference

### 6.3 Metrics

- **Primary:** Average Precision (AP) on Peptides-func (multi-label classification)
- **Secondary:** Training/inference time, parameter count, memory usage
- **Diagnostic:** Proxy embedding statistics (norms, pairwise distances, MMD to node embeddings), attention weight entropy, effective graph diameter with proxies

### 6.4 Additional Datasets (for generalization)

- Peptides-struct (LRGB) — graph regression, same molecular domain
- PascalVOC-SP, COCO-SP (LRGB) — node classification on superpixel graphs
- TreeNeighborsMatch — synthetic long-range task
- ZINC — molecular property prediction (smaller, good for quick iteration)

---

## 7. Potential Pitfalls and Mitigations

### 7.1 Optimization Landscape for Proxy Targets

**Risk:** The loss surface w.r.t. free proxy embeddings (Stage 2) may be highly non-convex, leading to poor local minima or instability.

**Mitigations:**
- Multiple random restarts (e.g., 5–10 per graph), keep the best.
- Warm-start from informative initializations (e.g., cluster centroids of node embeddings via k-means, or random node embeddings from the same graph).
- Use Adam with learning rate warmup and cosine decay.
- Gradient clipping on the proxy embeddings.

### 7.2 Flow Matching Model Overfitting

**Risk:** With a limited number of training graphs (~15,535 in Peptides-func), the Transformer denoiser may overfit.

**Mitigations:**
- Keep the denoiser small (2–4 layers, narrow hidden dimension).
- Use dropout and weight decay aggressively.
- Data augmentation: multiple restarts in Stage 2 give multiple valid targets per graph, effectively multiplying the dataset.
- Consider a simpler architecture (MLP with global graph conditioning) if the Transformer denoiser overfits.

### 7.3 Distribution Mismatch

**Risk:** Generated proxy embeddings may have different statistics than what the pretrained transformer expects.

**Mitigations:**
- Train the flow matching model in standardized space (see Section 3.5.6).
- Apply frozen LayerNorm to generated proxies.
- Monitor embedding norms and cosine similarities.

### 7.4 One-Step Generation Quality

**Risk:** One Euler step may not produce high-enough quality proxy embeddings, especially if the learned vector field is complex.

**Mitigations:**
- Use 2–4 Euler steps at inference (memory cost is negligible outside of end-to-end finetuning).
- During end-to-end finetuning, use 1 step for gradient efficiency; at evaluation, use more steps.
- Consider distilling the multi-step model to a single-step model if there's a large quality gap.

### 7.5 Marginal Gains Don't Justify Complexity

**Risk:** The flow matching pipeline may only marginally outperform simpler baselines.

**Mitigations:**
- Establish baselines early (see companion document).
- Emphasize generalization analysis: show proxies vary meaningfully across graphs.
- Demonstrate failure modes of simpler methods on specific graph types.

### 7.6 Training Pipeline Complexity

**Risk:** A multi-stage pipeline is hard to tune, reproduce, and debug.

**Mitigations:**
- Implement and validate each stage independently.
- Log extensively: loss curves, embedding statistics, attention maps, per-class AP.
- Use configuration management (Hydra) and experiment tracking (W&B).

---

## 8. Implementation Notes

### 8.1 Peptides-func Specifics

- **Dataset size:** ~15,535 graphs (train/val/test split provided by LRGB).
- **Task:** 10-class multi-label binary classification.
- **Metric:** Average Precision (AP).
- **Graph size:** Typically 100–300 nodes per graph.
- **Known behavior:** Graph transformers overfit in 100–200 epochs. Use early stopping on validation AP with patience ~20–30 epochs.

### 8.2 Suggested Hyperparameter Starting Points

```yaml
# Stage 0/1: Pretrained Transformer
model: GPS
hidden_dim: 64
num_layers: 4-6
num_heads: 8
dropout: 0.1-0.2
lr: 1e-3
weight_decay: 1e-5
early_stopping_patience: 25
max_epochs: 300

# Stage 2: Proxy Optimization
num_proxies_M: 8
proxy_lr: 1e-2
proxy_optimizer: Adam
num_iterations: 500
mmd_lambda: 0.05
num_restarts: 5
gradient_clip: 1.0
init_strategy: randn_matched

# Stage 3: Flow Matching Model
denoiser_layers: 3
denoiser_heads: 4
denoiser_dim: 64
denoiser_lr: 1e-4
denoiser_dropout: 0.15
time_sampling: uniform  # or logit_normal
euler_steps_train: N/A  # flow matching trains on single-step regression
euler_steps_inference: 1  # try 1, 2, 4
batch_size: 64
max_epochs: 200

# Stage 4: End-to-End Fine-tuning
finetune_lr_transformer: 1e-5
finetune_lr_denoiser: 1e-4
euler_steps_finetune: 1  # must be 1 for memory
finetune_epochs: 50
early_stopping_patience: 15
```

### 8.3 Computational Budget Estimate

| Stage | Approximate Time (1× A100) |
|---|---|
| Stage 0: Validation gate | 1–3 hours |
| Stage 1: Pretrain transformer | 2–6 hours |
| Stage 2: Optimize proxies (all training graphs) | 4–12 hours |
| Stage 3: Train flow matching model | 2–4 hours |
| Stage 4: End-to-end finetuning | 2–6 hours |
| **Total** | **~11–31 hours** |

### 8.4 Code Structure

```
project/
├── configs/
├── data/
├── models/
│   ├── transformer.py         # Graph transformer (GPS, etc.)
│   ├── denoiser.py            # Transformer denoiser with cross-attention
│   ├── flow_matching.py       # OT-CFM training & sampling
│   └── proxy_optimizer.py     # Stage 2 per-graph optimization
├── training/
│   ├── validate_premise.py    # Stage 0
│   ├── pretrain.py            # Stage 1
│   ├── optimize_targets.py    # Stage 2
│   ├── train_flow_matching.py # Stage 3
│   └── finetune.py            # Stage 4
├── evaluation/
│   ├── metrics.py
│   └── analysis.py
└── scripts/
    └── run_pipeline.sh
```

---

## 9. Open Questions and Future Directions

1. **Adaptive M:** Can we let the model decide how many proxies each graph needs, e.g., by generating M candidates and pruning those with low attention weight?

2. **Hierarchical proxies:** For very large graphs, could we generate proxies at multiple scales (local subgraph proxies → global proxies)?

3. **Transfer across datasets:** If proxy generation captures general structural principles, can a flow matching model trained on one molecular dataset transfer to another?

4. **Theoretical analysis:** Can we formalize the conditions under which M proxies reduce effective resistance (and hence oversquashing) in the augmented graph?

5. **Proxy interpretability:** Do generated proxies correspond to meaningful chemical substructures or functional groups in molecular graphs?

6. **Rectified flow refinement:** After initial flow matching training, use the rectified flow procedure (re-sample pairs from the learned coupling and retrain) to straighten the flow paths, improving one-step generation quality.

---

## 10. Summary: Minimal Viable Experiment

1. **Validate (Stage 0):** Optimize free proxy embeddings against a frozen transformer on a subset of training graphs. Confirm ≥5 AP point improvement on training data. **If this fails, stop.**
2. **Pretrain (Stage 1):** GPS on Peptides-func with early stopping → baseline AP.
3. **Optimize targets (Stage 2):** M=8 proxies per training graph, 5 restarts, MMD regularization.
4. **Train flow matching (Stage 3):** 3-layer Transformer denoiser, OT-CFM, one-step inference.
5. **Evaluate:** Generate proxies for val/test graphs, measure AP. Compare against K-fixed-virtual-nodes to quantify the value of graph-conditional generation.
6. **(Optional) Fine-tune (Stage 4):** End-to-end with one-step flow matching generation. Monitor for overfitting.

If this pipeline shows gains over both the baseline transformer and fixed virtual nodes, proceed to ablations, edge prediction, GNN integration, and additional datasets.