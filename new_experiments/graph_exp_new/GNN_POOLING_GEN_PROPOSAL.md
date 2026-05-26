## 3. Method

### 3.0 Pipeline Overview
```
Single-stage end-to-end training:
  Node Features → Embedding Layer → GNN Proxy Encoder (multi-hop, multi-pool → decode → M proxies)
                                  → concat [X; B] → Transformer Layers → Readout → Classification

All components trained jointly from scratch with task loss.
```

The architecture is a standard graph transformer augmented with a **GNN-based proxy encoder** inserted between the embedding layer and the transformer layers. The encoder runs a lightweight GNN over the graph, builds a multi-scale multi-pool graph descriptor, decodes M proxy embeddings from it via a shared MLP conditioned on proxy index embeddings, and concatenates them with the original node embeddings for the transformer to process.

---

### 3.1 Architecture

The full model has three components trained jointly:

#### 3.1.1 Embedding Layer

Standard node/edge feature embedding: AtomEncoder / BondEncoder / nn.Embedding → X ∈ ℝ^{N×d}

Same as the baseline graph transformer.

#### 3.1.2 GNN Proxy Encoder

**Note:** Keep this module a separate class in a separate file. The input is node embeddings X ∈ ℝ^{N×d} and edge_index (and optionally edge features). The output is proxy embeddings B ∈ ℝ^{M×d}.

##### Step 1 — Multi-hop GNN forward pass

Run a K-layer GNN over the graph, collecting intermediate representations at every layer:
```
H⁰ = X ∈ ℝ^{N×d}                                    (0-hop: raw node embeddings)
H¹ = GNN_Layer_1(H⁰, edge_index) ∈ ℝ^{N×d}          (1-hop neighborhood)
H² = GNN_Layer_2(H¹, edge_index) ∈ ℝ^{N×d}          (2-hop neighborhood)
...
Hᴷ = GNN_Layer_K(Hᴷ⁻¹, edge_index) ∈ ℝ^{N×d}       (K-hop neighborhood)
```

Each GNN layer applies message passing + nonlinearity + residual connection + LayerNorm:
```
Hᵏ = LayerNorm(Hᵏ⁻¹ + σ(MessagePassing(Hᵏ⁻¹, edge_index)))
```

The GNN type is configurable (GCN, GIN, GINE, GAT). For Peptides-func, GINE (GIN with edge features) is recommended since bond type information is available. Keep the GNN in the same d-dimensional space as the transformer.

**On GNN depth:** K = 3–5 is the practical range. Beyond ~5 layers, oversmoothing causes node representations to converge even with residual connections. For Peptides-func (100–300 nodes, typical diameter ~15–25), K=4 already covers a significant portion of the graph at the deepest level. Default K=4 (giving 5 hop levels), ablate over {2, 3, 4, 5}.

This gives K+1 sets of node representations: {H⁰, H¹, ..., Hᴷ}.

##### Step 2 — Multi-scale multi-pool graph descriptor

Apply P pooling functions to the node representations at each hop level, then concatenate everything into a single graph descriptor vector:

**Pooling functions** (configurable, default all three):
```
mean_pool(Hᵏ) = (1/N) Σᵢ hᵢᵏ                              ∈ ℝ^d
max_pool(Hᵏ)  = max_i(hᵢᵏ)        (element-wise max)       ∈ ℝ^d
std_pool(Hᵏ)  = sqrt((1/N) Σᵢ (hᵢᵏ - mean)²)              ∈ ℝ^d
```

(For batched graphs with padding, pool over unmasked nodes only.)

**Concatenate across all hop levels and all pool types:**
```
g = [ mean(H⁰) ; max(H⁰) ; std(H⁰) ;
      mean(H¹) ; max(H¹) ; std(H¹) ;
      ...
      mean(Hᴷ) ; max(Hᴷ) ; std(Hᴷ) ]  ∈ ℝ^{D}

where D = P · (K+1) · d
```

With P=3 (mean, max, std), K=4, d=128: D = 3 × 5 × 128 = 1920.
With P=3, K=4, d=256: D = 3 × 5 × 256 = 3840.

Each component of g captures a different structural scale (hop depth) and a different statistical aspect (central tendency, extremes, spread) of the graph. This is the graph's multi-resolution structural fingerprint.

##### Step 3 — Decode M proxy embeddings

**The scaling problem:** A single linear layer from g to M·d outputs would have D × M·d parameters. For D=3840, M=128, d=256, that's ~126M parameters in one layer — impractical.

**Solution: Shared MLP with proxy index embeddings.** Instead of decoding all M proxies at once, use a single shared MLP that decodes one proxy at a time, conditioned on a learnable proxy index embedding that tells the MLP *which* proxy to produce. The MLP is called M times (batched along the M dimension), and its size is independent of M.

**Proxy index embeddings:**
```
E ∈ ℝ^{M×d_idx}       (learnable lookup table, d_idx is small, e.g. 32 or 64)
```

These are shared across graphs but are **not** the proxy embeddings themselves — they are conditioning signals analogous to positional encodings. The actual proxy output is fully determined by g (which is per-graph). The index embedding just tells the MLP "produce the proxy assigned to slot m."

**Shared decoder MLP:**

For each proxy slot m (batched in parallel across M):
```
input_m = [g ; e_m] ∈ ℝ^{D + d_idx}
b_m = MLP(input_m) ∈ ℝ^d
```

where MLP is multiple linear layers:
```
Linear(D + d_idx, h₁) → GELU → Dropout →
Linear(h₁, h₂) → GELU → Dropout →
Linear(h₂, d)
```

Default h₁ = h₂ = 4·d (configurable). For d=256: layers are (D+d_idx → 1024 → 1024 → 256). Each layer is a modest-sized linear transform.

**Batched forward pass:**
```
g_repeated = g.unsqueeze(0).expand(M, -1)    ∈ ℝ^{M×D}       (same descriptor for all proxies)
inputs = cat([g_repeated, E], dim=-1)         ∈ ℝ^{M×(D+d_idx)}
B = MLP(inputs)                               ∈ ℝ^{M×d}
```

In a batched setting (batch size B_s, variable N per graph):
```
g_repeated: ℝ^{B_s × M × D}
E_repeated: ℝ^{B_s × M × d_idx}       (same E for all graphs)
inputs:     ℝ^{B_s × M × (D+d_idx)}
B:          ℝ^{B_s × M × d}
```

This is just a batched MLP call — trivially parallelizable on GPU.

**Why this scales:** The MLP parameter count is fixed: (D + d_idx) × h₁ + h₁ × h₂ + h₂ × d. Increasing M only adds M × d_idx parameters for the index embeddings (e.g. 128 × 64 = 8192 extra params for M=128 — negligible). Going from M=4 to M=128 does not change the MLP at all.

**Alternative: Grouped decoding.** Instead of a shared MLP + index embeddings, split the M proxies into G groups. Each group has its own smaller MLP that decodes M/G proxies:
```
For group g_idx = 1..G:
    B_group = reshape(MLP_g_idx(g), [M/G, d])    ∈ ℝ^{(M/G)×d}
B = cat([B_1, ..., B_G], dim=0)                   ∈ ℝ^{M×d}
```

Each group MLP: Linear(D, h) → GELU → Linear(h, (M/G)·d). For G=8, M=128, d=256: output per group is 16×256 = 4096, so the last layer is h→4096 — manageable.

The shared MLP approach is preferred (cleaner scaling, fewer total parameters, single code path). Grouped decoding is available as an ablation. Both are configurable — select via input parameter `decode_mode ∈ {shared, grouped}`.

##### Summary of shapes
```
Input:      X ∈ ℝ^{N×d}, edge_index
GNN:        H⁰...Hᴷ, each ∈ ℝ^{N×d}               (K+1 hop levels)
Pools:      P vectors per hop level, each ∈ ℝ^d      (P·(K+1) vectors total)
Descriptor: g ∈ ℝ^{D}, D = P·(K+1)·d                (multi-scale graph fingerprint)
Index:      E ∈ ℝ^{M×d_idx}                          (proxy index embeddings)
Decode:     B ∈ ℝ^{M×d}                              (M proxies)
```

**OTHER STEPS REMAIN SAME AS THE PIPELINE THIS IS INTEGRATED IN**