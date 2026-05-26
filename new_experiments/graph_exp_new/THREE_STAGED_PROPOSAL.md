## 3. Revised Pipeline

### Stage 1: Pretrain Graph Transformer (UNCHANGED)
Train a plain graph transformer on Peptides-func. Same as FLOW_MATCHING_PROPOSAL Stage 1. Freeze after training.

### Stage 2: Train Proxy Generator on Task Loss (NEW — replaces old Stages 2+3)

**Setup:**
- Freeze the pretrained transformer (all parameters).
- Attach a proxy generator module.
- Train the generator to minimize task loss by producing proxies that, when inserted into the frozen transformer, improve predictions.

**Training loop (per batch):**
```
X_dense, mask = frozen_transformer.encode_dense(batch)  # no grad through encoder
proxies = generator(X_dense, mask)                        # generator has grad
logits, _ = frozen_transformer(batch, proxy_embeddings=proxies,
                                precomputed_dense=(X_dense.detach(), mask))
loss = BCE(logits, labels)
loss.backward()  # gradients flow only into generator
optimizer.step()
```

**Key detail:** The node embeddings X used as input to the generator should be detached from the transformer's computation graph, so that only the generator parameters are updated. The transformer's forward pass (attention layers + readout) remains in the graph for gradient flow through the proxy embeddings.

**Evaluation:** Same as before — val AP with generated proxies inserted into frozen transformer. Early stop on val AP, patience 30.

### Stage 3: End-to-End Fine-tuning (MODIFIED — was Stage 4)

After the generator has converged to a good solution with the frozen transformer:

1. **Phase A (K=20 epochs):** Keep transformer frozen, continue training generator with a reduced learning rate (0.1x). This stabilizes the generator before the landscape shifts.
2. **Phase B:** Unfreeze transformer at 0.1x the generator's learning rate. Both adapt, but the transformer moves slowly while the generator adjusts to the shifting landscape.
3. **Proxy dropout (p=0.3):** During fine-tuning, randomly drop all proxies for 30% of batches. This prevents catastrophic forgetting of the proxy-free capability and forces proxies to provide additive value.

Early stop on val AP, patience 20.

### Stage 4: GNN Integration (UNCHANGED, OPTIONAL)
Same as before. Only attempt after Stage 3 converges cleanly.

---

## 4. Generator Architectures to Try

The new Stage 2 is generator-agnostic. Any module that takes (B, N, d) node embeddings + mask and outputs (B, M, d) proxy embeddings can be plugged in. The existing `BaseGenerator` interface already supports this.

**Priority order based on likelihood of success:**

### 4.1 Score-Based Generator (HIGHEST PRIORITY — already implemented)

The `ScoreBasedGenerator` from SCORE_BASED_GEN_PROPOSAL / generators.py is the best starting point. It's already implemented, deterministic, and structurally sound:

```
V = Linear(X)                    # value projection
S = MLP(X) -> (B, N, M)          # M score vectors per node
A = softmax(S, dim=nodes)        # soft selection weights
B0 = A^T @ V                     # weighted aggregation -> (B, M, d)
B = SelfAttnRefinement(B0)       # proxy self-attention layers
```

**Change from current implementation:** Remove the MMD loss computation against targets entirely. The generator's only loss is the task loss flowing back through the frozen transformer. The `aux_loss` return should be None — the task loss IS the training signal.

**Why this should work now when end-to-end didn't:** The end-to-end failure was a bootstrapping problem (transformer + generator both untrained simultaneously). With a frozen pretrained transformer, the generator gets a stable, meaningful gradient signal from the very first step. The transformer already knows how to use good token representations; the generator just needs to learn what representations help.

### 4.2 PMA-Style Cross-Attention with Per-Graph Queries (new)

The Set Transformer PMA approach failed before because global learned queries Q ∈ R^{M x d} are identical for every graph. The fix is to make Q per-graph while maintaining diversity.

**Architecture:**

```python
class PMAGenerator(BaseGenerator):
    """
    Pooling by Multihead Attention with per-graph query generation.

    Step 1: Generate M diverse per-graph query seeds
    Step 2: Cross-attention from queries to node embeddings
    Step 3: Optional self-attention refinement among proxies
    """
    def __init__(self, num_proxies, input_dim, num_heads=4,
                 num_layers=2, dropout=0.2, query_mode="farthest_point"):
        super().__init__(num_proxies, input_dim)
        self.query_mode = query_mode

        # Cross-attention: queries attend to nodes
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(input_dim),
                "norm_kv": nn.LayerNorm(input_dim),
                "cross_attn": nn.MultiheadAttention(
                    input_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "norm_ff": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim * 4, input_dim),
                ),
            }))

    def _get_query_seeds(self, node_embeddings, mask):
        """
        Generate M diverse per-graph query seed vectors.
        These are DETACHED from the computation graph — they serve only
        as initialization for the cross-attention queries.
        """
        B, N, d = node_embeddings.shape
        M = self.num_proxies

        with torch.no_grad():
            if self.query_mode == "farthest_point":
                # Greedy farthest-point sampling: deterministic, maximizes diversity
                seeds = []
                for b in range(B):
                    valid = node_embeddings[b][mask[b]]  # (n_valid, d)
                    n_valid = valid.shape[0]
                    if n_valid <= M:
                        # Pad with duplicates if fewer nodes than proxies
                        idx = torch.arange(n_valid, device=valid.device)
                        idx = idx.repeat(M // n_valid + 1)[:M]
                    else:
                        idx = [torch.randint(n_valid, (1,)).item()]
                        for _ in range(M - 1):
                            dists = torch.cdist(valid[idx], valid).min(dim=0).values
                            idx.append(dists.argmax().item())
                        idx = torch.tensor(idx, device=valid.device)
                    seeds.append(valid[idx])  # (M, d)
                return torch.stack(seeds).detach()  # (B, M, d)

            elif self.query_mode == "soft_kmeans":
                # Soft k-means clustering on node embeddings
                seeds = []
                for b in range(B):
                    valid = node_embeddings[b][mask[b]]
                    seeds.append(self._soft_kmeans(valid, M))
                return torch.stack(seeds).detach()  # (B, M, d)

    @staticmethod
    def _soft_kmeans(X, K, n_iters=3, temp=1.0):
        n = X.shape[0]
        if n <= K:
            idx = torch.arange(n, device=X.device).repeat(K // n + 1)[:K]
            return X[idx]
        idx = []
        idx.append(torch.randint(n, (1,)).item())
        for _ in range(K - 1):
            dists = torch.cdist(X[idx], X).min(dim=0).values
            idx.append(dists.argmax().item())
        centers = X[torch.tensor(idx, device=X.device)]
        for _ in range(n_iters):
            dists = torch.cdist(X, centers)
            weights = F.softmax(-dists / temp, dim=1)
            centers = (weights.T @ X) / (weights.sum(0, keepdim=True).T + 1e-8)
        return centers

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        # Step 1: Get per-graph diverse query seeds (detached)
        queries = self._get_query_seeds(node_embeddings, mask)  # (B, M, d)

        # Step 2-3: Cross-attention layers
        key_padding_mask = ~mask  # True = padding for MHA
        x = queries
        for layer in self.cross_attn_layers:
            # Cross-attention: queries attend to nodes
            q_normed = layer["norm_q"](x)
            kv_normed = layer["norm_kv"](node_embeddings)
            attn_out, _ = layer["cross_attn"](
                q_normed, kv_normed, kv_normed,
                key_padding_mask=key_padding_mask
            )
            x = x + attn_out
            # FFN
            x = x + layer["ffn"](layer["norm_ff"](x))

        return x, None  # No aux_loss — task loss is the only signal
```

**Why this addresses the Set Transformer failure:**
- Queries are per-graph (derived from that graph's node embeddings), not global
- Farthest-point sampling guarantees diversity (queries are maximally spread in embedding space)
- Detaching seeds means gradients flow through the cross-attention parameters, not through the seed selection — the attention learns what to extract from nodes given diverse starting points
- The cross-attention parameters ARE shared across graphs (generalization), but the queries they operate on are graph-specific (adaptation)

**query_mode options (configurable):**
- `farthest_point` (recommended default): Deterministic, maximally diverse, O(NM) per graph
- `soft_kmeans`: Slightly more principled clustering, but introduces randomness from initialization and is slower

### 4.3 Graph Coarsening Generator (new)

Learnt graph coarsening (DiffPool/MinCutPool style) is a natural fit for proxy generation. The idea: learn a soft assignment matrix that maps N nodes to M "super-nodes" (proxies), using the graph's actual topology.

**Architecture:**

```python
class GraphCoarseningGenerator(BaseGenerator):
    """
    Generates proxies via learnt graph coarsening.

    Uses a GNN to compute soft assignment of N nodes to M clusters,
    then aggregates node features per cluster to produce proxy embeddings.
    Optionally applies MinCut or Ortho regularization to encourage
    meaningful, non-degenerate clusters.

    This naturally respects graph topology: connected nodes tend to be
    assigned to the same cluster, producing proxies that summarize
    local structural neighborhoods.
    """
    def __init__(self, num_proxies, input_dim, gnn_layers=2,
                 gnn_type="GIN", dropout=0.2, reg_type="mincut",
                 reg_weight=0.1, num_refine_layers=1, num_heads=4):
        super().__init__(num_proxies, input_dim)
        self.reg_type = reg_type
        self.reg_weight = reg_weight

        # GNN for computing assignment logits
        # Input: node embeddings (N, d)
        # Output: assignment logits (N, M)
        self.assign_gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            if gnn_type == "GIN":
                from torch_geometric.nn import GINConv
                gin_nn = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim),
                )
                self.assign_gnn_layers.append(GINConv(gin_nn))
            elif gnn_type == "GCN":
                from torch_geometric.nn import GCNConv
                self.assign_gnn_layers.append(GCNConv(input_dim, input_dim))

        self.assign_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(gnn_layers)
        ])

        # Final projection to M assignment logits
        self.assign_proj = nn.Linear(input_dim, num_proxies)

        # Optional refinement: self-attention among coarsened proxies
        self.refinement_layers = nn.ModuleList()
        for _ in range(num_refine_layers):
            self.refinement_layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(input_dim),
                "attn": nn.MultiheadAttention(
                    input_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "norm2": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim * 4, input_dim),
                ),
            }))

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        """
        For this generator, node_embeddings should be FLAT (total_N, d),
        not dense-batched, since it needs edge_index for GNN.
        kwargs must contain 'edge_index', 'batch_vec'.
        Returns dense (B, M, d) proxy embeddings.
        """
        edge_index = kwargs["edge_index"]
        batch_vec = kwargs["batch_vec"]
        num_graphs = int(batch_vec.max().item()) + 1

        # GNN forward for assignment computation
        h = node_embeddings
        for gnn_layer, norm in zip(self.assign_gnn_layers, self.assign_norms):
            h = norm(h + F.relu(gnn_layer(h, edge_index)))

        # Assignment logits and soft assignment
        assign_logits = self.assign_proj(h)  # (total_N, M)
        S = F.softmax(assign_logits, dim=-1)  # (total_N, M) — soft assignment

        # Aggregate node features per cluster, per graph
        # Use scatter to handle variable-size graphs
        # S^T @ X per graph -> (M, d) per graph
        proxies_list = []
        for g in range(num_graphs):
            g_mask = (batch_vec == g)
            S_g = S[g_mask]        # (n_g, M)
            X_g = node_embeddings[g_mask]  # (n_g, d)
            # Normalize: each cluster's weights sum to 1
            S_g_norm = S_g / (S_g.sum(dim=0, keepdim=True) + 1e-8)
            proxies_g = S_g_norm.T @ X_g  # (M, d)
            proxies_list.append(proxies_g)

        proxy_embeddings = torch.stack(proxies_list)  # (B, M, d)

        # Refinement layers
        x = proxy_embeddings
        for layer in self.refinement_layers:
            normed = layer["norm1"](x)
            attn_out, _ = layer["attn"](normed, normed, normed)
            x = x + attn_out
            x = x + layer["ffn"](layer["norm2"](x))
        proxy_embeddings = x

        # Compute coarsening regularization loss
        aux_loss = None
        if self.reg_type == "mincut" and self.reg_weight > 0:
            # MinCut regularization: encourages clusters to respect graph cuts
            # L_mincut = -Tr(S^T A S) / Tr(S^T D S) + ortho_reg
            # Simplified: just orthogonality loss to prevent all nodes -> 1 cluster
            ortho_losses = []
            for g in range(num_graphs):
                g_mask = (batch_vec == g)
                S_g = S[g_mask]  # (n_g, M)
                # Orthogonality: (S^T S / n) should be close to I/M
                StS = (S_g.T @ S_g) / S_g.shape[0]
                I_M = torch.eye(self.num_proxies, device=S_g.device) / self.num_proxies
                ortho_losses.append(torch.norm(StS - I_M))
            aux_loss = self.reg_weight * torch.stack(ortho_losses).mean()

        return proxy_embeddings, aux_loss
```

**Why graph coarsening is worth trying:**
- Topology-aware: proxies summarize structurally coherent subgraphs, not arbitrary node combinations
- Natural diversity: the assignment GNN is incentivized by MinCut/ortho regularization to produce non-overlapping clusters
- Per-graph by construction: assignments depend on the specific graph's structure and features
- Established theoretical grounding (spectral clustering, graph partitioning)

**Potential issues:**
- Requires edge_index, so needs the flat (not dense-batched) interface — slightly different data flow than the score-based and PMA generators
- The per-graph loop in the forward pass is not ideal for GPU parallelism (can be optimized with scatter operations later)
- MinCut regularization adds a hyperparameter; may fight the task loss if cluster structure doesn't align with task-relevant structure

### 4.4 GNN Pooling Generator (already implemented)

The `GNNPoolingGenerator` pools graph features into a single descriptor vector g, then decodes M proxies via a shared MLP. The problem: all graph-level information is compressed into one vector before proxy generation. This bottleneck means proxies can only differ via the index embedding, which provides very weak per-proxy specialization signal.

**Keep for ablation, but don't expect it to outperform 4.1-4.3.** The fundamental issue is that global pooling destroys the node-level information that proxies need to specialize.

### 4.5 Flow Matching Generator (keep for comparison only)

With the new Stage 2 (task loss training), flow matching is no longer needed as the primary generator. However, it could still be used as a baseline comparison in the paper. **Do not invest further development time on flow matching until the deterministic generators are properly evaluated.**

---

## 6. Implementation Plan

### Phase 1: Score-Based Generator with Task Loss

**Changes to existing code:**

1. **Make `main_three_staged.py` with Stage 3 training loop modified from `main_staged.py`:**
   - Remove: loading of Stage 2 optimized targets
   - Remove: flow matching / MMD reconstruction loss
   - Add: forward pass through frozen transformer with generated proxies
   - Training signal: `task_loss = BCE(transformer_output, labels)` only
   - Optimizer updates only generator parameters
   - Evaluate by generating proxies -> inserting into frozen transformer -> computing val/test AP
   - This is already roughly in place for the flow matching eval; adapt it

2. **From `ScoreBasedGenerator.forward()`:**
   - The MMD loss lambda will be passed as 0 in `run_three_staged.sh`

3. **Add proxy diagnostics (new, add to training loop):**
   - Per-epoch: log proxy pairwise cosine similarity (detect Mode B collapse)
   - Per-epoch: log attention weight entropy from the score MLP (detect virtual-node averaging)
   - Per-epoch: log generator gradient norm (detect Mode A starvation)
   - Sanity check every epoch (after validation): replace generated proxies with their mean -> re-evaluate AP. If AP doesn't drop, proxies are inert.

**Expected outcome:** This should work better than the flow matching pipeline because the generator gets direct task supervision. If it doesn't outperform vanilla transformer, the proxy idea itself may have limited value on Peptides-func — that would be important to know early.

### Phase 2: PMA Generator with Per-Graph Queries

1. **Add `PMAGenerator` class to `generators.py`** (architecture in Section 4.2)
2. **Make generator type a CLI argument** in `main_staged.py` and `main_three_staged.py` and `main_e2e.py`

**Ablations specific to PMA:**
- `query_mode`: farthest_point vs. soft_kmeans
- Number of cross-attention layers: {1, 2, 3}
- Whether to include self-attention among proxies after cross-attention

### Phase 3: Graph Coarsening Generator

1. **Add `GraphCoarseningGenerator` class to `generators.py`** (architecture in Section 4.3)
2. **Handle the interface difference:** this generator needs flat node embeddings + edge_index, not dense-batched but flat ones can be converted from `dense_x` using `dense_mask` and any other processing needed. Refer `vanilla_gt.py`.
