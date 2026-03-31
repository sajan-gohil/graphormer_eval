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

### 2.4 Empirical Validation (Completed)

Baseline experiments have confirmed the premise:

- **Fixed/mean-based proxies** (virtual nodes, k-means centroids, random node sampling, fixed learnable tokens) give similar or slightly worse results than the vanilla transformer. Per-graph conditioning is necessary.
- **Optimized free proxy embeddings** (gradient-optimized against frozen transformer) push training AP to nearly 100%, confirming massive headroom for proxy-based improvement.
- **MMD regularization** at λ=0.01 effectively constrains proxy distribution.
- **Both attention routing modes** — full self-attention (N+M × N+M) and routed cross-attention (N→M→N) — perform comparably. Either can be used.

The gap between fixed proxies (no improvement) and optimized proxies (~100% train AP) confirms that **graph-conditional generation is essential** — the value lies in producing proxies tailored to each graph, not in having additional tokens per se.

---

## 3. Method

### 3.0 Pipeline Overview

```
Stage 1: Pretrain Graph Transformer (frozen after this stage)
Stage 2: Optimize proxy embeddings per graph -> These become targets which are produced by next phase
Stage 3: Train conditional flow matching model to generate proxy embeddings
         ↳ Evaluated by downstream val AP, not just reconstruction loss
Stage 4: End-to-end fine-tuning of full pipeline
Stage 5: Edge prediction and GNN integration
```

---

### 3.1 Stage 1 — Pretrain Graph Transformer

Train a plain graph transformer on Peptides-func. The architecture is: node feature embedding → stack of Transformer layers (multi-head self-attention + FFN with residual connections and LayerNorm) → global readout (mean pool) → classification head (MLP → 10 outputs, sigmoid). There is no local message passing component — attention is the only inter-node operation.

Train a standard graph transformer on Peptides-func. This transformer serves two purposes:
- It provides the **embedding space** in which proxy nodes will live.
- It provides the **frozen evaluation function** used to optimize proxy targets in Stage 2.

Read `vanilla_gt.py` for sample implementation and how data is loaded and batched with `to_dense_batch` and `dense_mask`.

#### 3.1.1 Overfitting Mitigation

The base transformer overfits quickly on Peptides-func (typically within 100–200 epochs). The following mitigations should be applied:

**Regularization:**
- **Dropout:** Apply dropout at 0.3 on attention weights, FFN intermediate activations, and the embedding layer. This is the single most impactful regularizer for small graph datasets.
- **Early stopping:** Patience of 5 epochs

- **Gradient clipping:** Clip gradient norm at 1.0 to stabilize training and prevent large updates in late training.

### 3.2 Stage 2 — Optimize Proxy Embeddings (Target Generation)

**Goal:** For each training graph, find M proxy embeddings that, when inserted into the frozen pretrained transformer, improve the task loss. These optimized embeddings become the **regression targets** for the flow matching model.

Start with learnable tensors initialized randomly. With a frozen transformer from phase 1, merge the learnable tensors with other node embeddings coming from the embedding table and optimize them using task loss.

The proxy nodes are different for each graph. Run the optimization for fixed amount of iterations (number of iterations taken as input parameter). In each optimization cycle, save the best embeddings which produce the lowest loss for a graph. Save only if the overall loss of the graph reduces below a threshold or if the prediction of atleast one class which was incorrect before becomes correct. Start with `num_steps` (input param) number of steps, if after that desired optimization has not been achieved, continue for another `num_steps` steps.

The optimized proxy vectors should be saved per graph but the optimization should happen in batches. Read `optimization_phase_sample.py` for sample implementation.

This optimization should happen such that the distribution of optimized embeddings remains similar to embeddings of original nodes, so that the variability in optimized embeddings of different graphs is not too high and can be generated in next phase. Use MMD loss for distribution matching and use it as a constraint in optimization. Refer `mmd.py` for sample implementation.

Generate **multiple proxies** for each graph by running the optimization cycles multiple times. Save only if  atleast one classification which was incorrect without proxies is correct with proxies, or the loss has reduced for the graph to a low enough threshold.

Calculate and log AP using the optimized embeddings for every 1000 samples.


After optimization, additionally filter:

- **Keep** samples where the per-graph task loss improved over the proxy-free baseline.
- **Keep** samples where at least one additional class is correctly predicted.
- **Discard** samples where MMD loss is more than 3 std deviations away from the distribution of distances of other node embeddings (calculate a distribution/distribution parameters of MMD loss for original embeddings beforehand.).


---

### 3.3 Stage 3 — Train Conditional Flow Matching Model

**Goal:** Train a generative model that, given the node embeddings X of any graph, produces M proxy embeddings B that improve the transformer's predictions on that graph. The targets while training will be a set of proxy embeddings learnt by optimization previously.

#### 3.3.1 Why Flow Matching

- **One-step generation at inference:** Flow matching learns a vector field that transports N(0,I) to the target distribution. A single Euler step suffices. Critical for Stage 4 end-to-end finetuning.
- **Simple training objective:** Direct regression on the conditional vector field — no noise schedules, no variance weighting.
- **Clean backpropagation:** Gradients flow through one function evaluation.
- **No mode coverage issues of VAEs:** No posterior collapse or mode averaging.

#### 3.3.2 Flow Matching Formulation

**Conditional Flow Matching (CFM)** defines a time-dependent vector field u_t that transports samples from p_0 = N(0, I) to p_1 (optimal proxy embeddings).

For target x_1 (an optimized proxy embedding set) and noise x_0 ~ N(0, I):

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

where v_θ is the learned vector field (Transformer denoiser), conditioned on node embeddings X.

**Sampling (inference):**
```
z_0 ~ N(0, I) ∈ ℝ^{M×d}
B = z_0 + v_θ(z_0, t=0 | X)          # One Euler step (t: 0→1)
```

Make number of Euler steps an input parameter with default as 1:
```
for t in [0, Δt, 2Δt, ..., 1-Δt]:
    z_{t+Δt} = z_t + Δt · v_θ(z_t, t | X)
B = z_1
```

#### 3.3.3 Architecture — Transformer Denoiser with Cross-Attention

```
Input:  x_t ∈ ℝ^{M×d}  (interpolated proxy tokens at time t)
Cond:   X ∈ ℝ^{N×d}     (original node embeddings, frozen, from the initial AtomEncoder/BondEncoder/nn.embedding layer.)
Time:   t ∈ [0, 1]       (flow time, embedded via sinusoidal or MLP encoding)

Time embedding: t_emb = MLP(sinusoidal(t)) ∈ ℝ^d

For each denoiser layer:
    x_t ← x_t + t_emb                        # Additive time conditioning
    x_t ← CrossAttention(Q=x_t, K=X, V=X)    # proxy tokens attend to node embeddings
    x_t ← FFN(x_t)

Output: v_θ(x_t, t | X) ∈ ℝ^{M×d}            # predicted vector field
```


#### 3.3.4 Downstream Evaluation to prevent Flow Matching Overfitting


Every epoch, run the following evaluation:
1. For each graph in the validation set, generate proxy embeddings using the current flow matching model (one Euler step).
2. Insert generated proxies into the frozen transformer.
3. Compute validation AP.
4. Log: flow matching train loss, flow matching val reconstruction loss, **downstream val AP**.
5. Save the checkpoint with the best downstream val AP.

Early stop on downstream val AP with patience 30 epochs.

Keep the denoiser architecture configurable (num layers, dropout percentage, num heads, hidden dim etc)

**Diagnostic: Flow matching loss vs downstream AP correlation.**

Plot flow matching reconstruction loss against downstream val AP across training. 

Similar to evaluation on validation data, infer and derive AP on test data as well.

---

### 3.4 Stage 4 — End-to-End Fine-Tuning

Fine-tune the full pipeline jointly:

```
Task Loss → Transformer → Flow Matching Generator → Embedding Layer
```

**Gradient flow with flow matching:** The generation step is a single forward pass:

```
B = z_0 + v_θ(z_0, t=0 | X)
```

This is fully differentiable. Gradients from the task loss flow through:
1. The transformer (producing ∂L/∂B)
2. Through the addition, into v_θ (producing ∂L/∂θ_denoiser)
3. Through the cross-attention in v_θ, into X (producing ∂L/∂X)
4. Into the embedding layer (producing ∂L/∂θ_embed)

No multi-step unrolling, no adjoint methods. This is the primary advantage of flow matching over DDPM.

**What gets updated during fine-tuning:**
- Denoiser parameters
- Transformer parameters
- Embedding layer

---

### 3.5 Stage 5 — Edge Prediction and GNN Integration

Instead of (or in addition to) inserting proxies into the transformer's attention, connect them to the original graph and use a GNN.

**Edge prediction options (input parameter to choose a mode):**

| Method | Description | Cost |
|---|---|---|
| Full connectivity | Connect each proxy to all N nodes | O(NM) edges — cheap for small M |
| Attention-weighted | Use final cross-attention weights from flow matching as edge weights | O(NM), soft, no extra parameters |
| Sigmoid dot-product | σ(Linear(b_i) · Linear(x_j)) for proxy i, node j | O(NM), learnable threshold |
| MLP classifier | MLP([b_i; x_j]) → {0,1} for each pair | O(NM) forward passes — feasible for small M |

**GNN integration:** Run a GNN on the augmented graph with proxy-to-node edges. This tests whether proxy quality transfers from the transformer setting to the GNN setting.

---

## 4. Inference Pipeline

At test time (for unseen graphs):

```
1. Compute node embeddings X via the embedding layer
2. Sample noise z_0 ~ N(0, I) ∈ ℝ^{M×d}
3. Generate proxy embeddings: B = z_0 + v_θ(z_0, t=0 | X)   [one Euler step]
4. Insert B into the transformer alongside X
5. Compute predictions
```

No per-graph optimization at inference.

---


## 5. Experimental Plan

### 5.1 Ablations

Make all these parameters configurable via input args for ablations later.

- **M (number of proxies):** M ∈ {2, 4, 8, 16, 32}
- **Attention routing:** Full self-attention (N+M) vs. routed cross-attention (N→M→N) — both work comparably, but ablate formally
- **MMD regularization:** Confirmed λ=0.01 is good; ablate λ ∈ {0, 0.005, 0.01, 0.05, 0.1}
- **Target optimization steps:** 50, 100, 200, 500, 1000 steps — identify generalization knee
- **Multiple target sets optimized per graph:** K ∈ {1, 3, 5, 10} — impact on flow matching generalization
- **Target noise injection:** std ∈ {0, 0.01, 0.03, 0.05}
- **End-to-end finetuning:** With and without Stage 4
- **Euler steps at inference:** 1 vs. 2 vs. 4
- **Ensemble sampling:** 1 vs. 5 vs. 10 proxy samples generated when evaluating and testing.

### 5.2 Metrics

Log these at appropriate intervals

- **Primary:** Average Precision (AP) on Peptides-func (multi-label classification) (every 1000 steps and after every epoch for validation and test sets)
- **Secondary:** Training/inference time, parameter count, memory usage (Every epoch)
- **Diagnostic:** Flow matching loss, downstream val AP correlation (plot of loss vs downstream AP) (Calculated considering every 1000 steps as new sample)

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
# Stage 1: Pretrained Transformer
model: GT
hidden_dim: 64
num_layers: 5
num_heads: 8
dropout: 0.3
weight_decay: 3e-4
batch_size: 256
lr: 1e-3
early_stopping_patience: 5
max_epochs: 500

# Stage 2: Proxy Optimization
num_proxies_M: 32
proxy_lr: 1e-2
proxy_optimizer: Adam
num_steps: 700
mmd_lambda: 0.05
num_restarts: 5-10
gradient_clip: 1.0
init_strategy: randn_matched

# Stage 3: Flow Matching Model
denoiser_layers: 4
denoiser_heads: 8
denoiser_dim: 128
denoiser_lr: 5e-4
denoiser_dropout: 0.2
weight_decay: 1e-4
target_noise_std: 0.02
time_sampling: uniform
batch_size: 256
max_epochs: 1000
eval_every: 5 epochs (downstream val AP)
early_stopping_patience: 20 (on downstream val AP)

# Stage 4: End-to-End Fine-tuning
finetune_lr_transformer: 1e-5
finetune_lr_denoiser: 1e-4
euler_steps: 1
finetune_max_epochs: 200
early_stopping_patience: 10
```

### 6.3 Code Structure

Keep the structure simple, no sub directories and new files for small modules/functions. All model classes in one file, data loading in one file and training loop for all phases in the main file along with input loading. No separate files or functions which set random seeds or setup logging. Minimal code focusing on functionality and accurate implementation.

For logging, use print statements with flush=True only. NO WANDB or MLFlow etc tools.

For config loading, keep an option to load from yaml files but main method should be command line args. Even if yaml config is passed, cli args should be overriding the input parameter.

---