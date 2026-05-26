# Alternate method for graph transformer improvement with proxies with deterministic generator for optimized proxies as targets

---
**PREVIOUS STEPS REMAIN SAME AS THE PIPELINE THIS IS INTEGRATED IN**

### 3.3 Stage 3 — Train Conditional Proxy Generator Model

**Note:** This Stage 3, section 3.3 might change, so keep this module a separate class in a separate file. The inputs and outputs will remain same - Original node embeddings and (learnt) target embeddings (optional, None by default for test generation). The output should be the matching loss and the new generated proxy embeddings.

**Goal:** Train a generative model that, given the node embeddings X of any graph, produces M proxy embeddings B that improve the transformer's predictions on that graph. The targets while training will be a set of proxy embeddings learnt by optimization previously.

#### 3.3.1 Why Non linear model

Linear combination of nodes were previously checked and they dod not perform well, hence learnt embeddings are required. We can do this by producing weights for combinations of original node embeddings, and then train the model to produce the optimized proxy embeddings we learnt in previous phase.


### 3.3.2 Architecture

X ∈ ℝ^{N×d} (node embeddings from the embedding layer), M proxies desired, d = hidden dim.

**Step 1 — Value projection.** Project node embeddings into a value space:

V = X · W_v, where W_v ∈ ℝ^{d×d}, so V ∈ ℝ^{N×d}

**Step 2 — Compute M score vectors.** A single shared MLP that outputs M scores per node. This is where the M "selectors" live — each output column is a different learned scoring function over nodes:

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

Stack steps 5–6 for L layers (input param) for more refinement. B₂ ∈ ℝ^{M×d} is the final set of proxy embeddings — fed into the pretrained transformer alongside X.

**For batching** (graphs have different N): use dense batching with padding and a mask, same as the other setup. mask out padding positions to −∞ before the softmax so padding nodes get zero weight.

The targets of this proxy generation model are the optimized embeddings learnt in phase 2, calculate the MMD loss as reconstruction/proxy matching loss.


**OTHER STEPS REMAIN SAME AS THE PIPELINE THIS IS INTEGRATED IN**