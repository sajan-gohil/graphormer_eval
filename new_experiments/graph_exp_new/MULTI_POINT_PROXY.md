# Multi-Point Proxy Insertion Plan

## 1. Motivation & Will It Help?

### Current Architecture Limitation

Today, proxy generation + cross-attention routing happens **once**, between the encoder output and the first transformer layer. The transformer stack then processes the refined node embeddings without any further proxy interaction. This means:

- Proxies see only the **initial** node representations (post-encoder, pre-transformer).
- The transformer layers refine node representations in isolation — the hierarchical abstraction that proxies provide is a one-shot injection.
- Later transformer layers cannot request "updated summaries" from proxies that reflect the evolving node state.

### Why Multi-Point Insertion Should Help

**1. Progressive Abstraction (strong argument for Peptides-func).**
Peptides-func is multi-label classification of peptide functions (10 classes, evaluated by AP). Molecular graphs have hierarchical structure: atoms → functional groups → secondary structure motifs → global function. A single proxy pass captures one level of abstraction. Interleaving proxy blocks at layers 0, 2, and 4 (in a 6-layer model) lets proxies progressively capture atom-level, substructure-level, and motif-level patterns — each pass sees richer node representations.

**2. Iterative Message Passing Through Bottleneck (analogy: Perceiver IO).**
Perceiver IO demonstrated that iterating cross-attention between a latent bottleneck and input tokens at multiple depths significantly improves performance over single-pass cross-attention. Our proxy mechanism is structurally identical: M latent proxies ↔ N input nodes. DeepMind's Perceiver IO paper showed that 6–8 cross-attention blocks with shared weights outperformed single-pass architectures across modalities. Our setting is directly analogous.

**3. Gradient Highway Effect.**
Inserting proxy cross-attention at intermediate points creates skip-connection-like gradient paths from the loss through the proxy router at layer k directly back to encoder outputs. This is similar to how DenseNet/U-Net skip connections help gradient flow. For the 10-label AP metric where some labels are rare, better gradient flow to early layers should help learn discriminative features for minority classes.

### Expected Impact on AP

Based on analogous architectures:
- Perceiver IO's iterative cross-attention: +2–5% over single-pass on classification tasks.
- U-Net skip connections in segmentation: often +5–10% over encoder-only.
- For Peptides-func (baseline AP ≈ 0.65–0.70 range), **+1–3 AP points** is a reasonable expectation.
- Risk: overfitting on the small Peptides-func dataset (15,535 graphs). Multi-point with separate generators adds parameters. Mitigate via shared generator (Option A).

### When It Might NOT Help
- If the graph transformer is very shallow (2 layers) — too few layers to split.
- If M is very large relative to N — the cross-attention cost dominates and the bottleneck effect is lost.
- If the generator is expensive (e.g., FlowMatchingGenerator with many ODE steps) and runs at every insertion point — training becomes prohibitively slow.


## 2. Architecture Design

### 2.1 Core Concept: Interleaved Transformer-Proxy Blocks

Replace the current sequential structure:

```
[Encoder] → [ProxyGen + Route @ 0] → [TF Layer 0] → [TF Layer 1] → ... → [TF Layer L-1] → [Readout]
```

With an interleaved structure (example: `--proxy_insertion_layers 0,2,4`):

```
[Encoder] → [ProxyGen + Route @ 0] → [TF Layer 0] → [TF Layer 1] → [ProxyGen + Route @ 2] → [TF Layer 2] → [TF Layer 3] → [ProxyGen + Route @ 4] → [TF Layer 4] → [TF Layer 5] → [Readout]
```

Insertion at layer `k` means: run proxy generation + routing **before** transformer layer `k`. The generator always receives the current node representations (output of layer k-1, or encoder output if k=0).

### 2.2 Insertion Point Semantics

Configurable via CLI flag:

```
--proxy_insertion_layers 0,2,4    # insert before layers 0, 2, and 4 (0-indexed)
```

Semantics:
- `-1` (or omitting the flag): **no proxy insertion at all** — proxies are disabled entirely.
- `0`: insert before layer 0 — between encoder and 1st transformer layer. This is equivalent to the current single-point behavior.
- `k` (for k > 0): insert before layer k — generator sees the output of layer k-1.
- `0,2,4`: insert at three points — before layers 0, 2, and 4.

### 2.3 Generator Strategy

**Option A: Shared Generator, Fresh Proxies (recommended default)**
- One generator instance, called at each insertion point with the current node embeddings.
- Proxies are regenerated fresh each time from the evolving node representations.
- Pro: minimal parameter overhead, generator sees progressively refined inputs.
- Con: generator must handle varying input distributions (early vs. late layer outputs).

**Option B: Separate Generators Per Insertion Point**
- Distinct generator instances for each insertion point.
- Pro: each generator specializes for its layer's representation quality.
- Con: multiplies generator parameters by number of insertion points.

**Recommendation:** Start with **Option A** (shared generator). Use **Option B** for ablation via `--separate_proxy_generators` flag.

### 2.4 Router Strategy: Shared vs. Separate

**Shared Router (default):** One CrossAttentionRouter instance used at all insertion points. Fewer parameters. Perceiver IO showed shared-weight cross-attention works well.

**Separate Routers:** Distinct router instances per insertion point. More expressive but more parameters.

**Recommendation:** Default to **shared router** with a `--separate_proxy_routers` flag for ablation.

### 2.5 N+M Concat Mode (No Changes)

When using the default N+M concatenation (i.e., `--use_cross_attn_routing` is NOT set), multi-point insertion simply concatenates M fresh proxies at each insertion point. This means tokens accumulate:

```
Layer 0 sees: N + M tokens (encoder output + proxies from insertion @ 0)
Layer 2 sees: N + 2M tokens (above + proxies from insertion @ 2)
Layer 4 sees: N + 3M tokens (above + proxies from insertion @ 4)
```

No special handling needed — each insertion point just concatenates M more proxy tokens and extends the attention mask with M more `True` entries. The readout still pools over the first N positions (for `nodes_only` scope) or all tokens (for `all_tokens` scope).

### 2.6 Aux Loss Aggregation

When the generator runs at multiple points, each invocation produces an aux_loss. Aggregation options:
- **Sum:** `total_aux = sum(aux_losses)` — treats all insertion points equally.
- **Weighted sum:** later insertions get lower weight (early representations need more guidance).

**Recommendation:** Simple sum, with optional `--proxy_aux_loss_decay` factor (e.g., 0.5 geometric decay per insertion point).


## 3. New Module: `MultiPointProxyWrapper`

A wrapper that encapsulates the generator(s) + router(s) and manages multi-point insertion logic. This keeps the backbone models clean.

```python
class MultiPointProxyWrapper(nn.Module):
    """
    Manages proxy generation and cross-attention routing at multiple
    insertion points within a transformer stack.

    Args:
        generator: BaseGenerator instance (used if shared).
        router: CrossAttentionRouter instance (used if shared, or None if concat mode).
        insertion_layers: sorted list of int — before which transformer layers
                          to inject proxy blocks (0-indexed).
        separate_generators: if True, create independent generator per insertion point.
        separate_routers: if True, create independent router per insertion point.
        aux_loss_decay: geometric decay factor for aux losses (1.0 = no decay).
    """
    def __init__(self, generator, router, insertion_layers,
                 separate_generators=False, separate_routers=False,
                 aux_loss_decay=1.0):
        super().__init__()
        self.insertion_layers = sorted(insertion_layers)
        self.aux_loss_decay = aux_loss_decay
        K = len(insertion_layers)

        # Generators
        if separate_generators:
            self.generators = nn.ModuleList([
                copy.deepcopy(generator) for _ in range(K)
            ])
        else:
            self.generators = nn.ModuleList([generator])

        # Routers (None if using N+M concat mode)
        if router is not None:
            if separate_routers:
                self.routers = nn.ModuleList([
                    copy.deepcopy(router) for _ in range(K)
                ])
            else:
                self.routers = nn.ModuleList([router])
        else:
            self.routers = None

    def get_generator(self, point_idx):
        if len(self.generators) == 1:
            return self.generators[0]
        return self.generators[point_idx]

    def get_router(self, point_idx):
        if self.routers is None:
            return None
        if len(self.routers) == 1:
            return self.routers[0]
        return self.routers[point_idx]

    def run_proxy_block(self, node_emb, mask, point_idx):
        """
        Run one proxy generation + routing pass.

        Args:
            node_emb: (B, N_current, d) current node/token embeddings.
            mask: (B, N_current) current mask.
            point_idx: index into self.insertion_layers.

        Returns:
            If router exists (cross-attn mode):
                refined_nodes: (B, N_current, d), updated_mask: same mask, aux_loss: scalar
            If no router (concat mode):
                augmented_tokens: (B, N_current + M, d), updated_mask: (B, N_current + M), aux_loss: scalar
        """
        gen = self.get_generator(point_idx)
        proxy_emb, aux_loss = gen(node_emb, mask)

        router = self.get_router(point_idx)
        if router is not None:
            # Cross-attention mode: N→M→N routing, token count unchanged
            refined = router(node_emb, proxy_emb, mask)
            return refined, mask, aux_loss
        else:
            # Concat mode: append M proxies, token count grows
            B, M, d = proxy_emb.shape
            aug_tokens = torch.cat([node_emb, proxy_emb], dim=1)
            aug_mask = torch.cat([
                mask,
                torch.ones(B, M, dtype=torch.bool, device=mask.device)
            ], dim=1)
            return aug_tokens, aug_mask, aux_loss
```


## 4. Required Code Changes

### 4.1 `generators.py`

**Add:** `MultiPointProxyWrapper` class as defined above. Needs `import copy` at top.

No changes to `CrossAttentionRouter` — in cross-attention mode the router returns refined node embeddings as before. In concat mode the router is not used (wrapper handles concatenation).

### 4.2 `models.py` — `GraphTransformer`

**Modified `__init__`:**
```python
def __init__(self, ..., cross_attn_router=None, multi_point_proxy=None):
    ...
    self.cross_attn_router = cross_attn_router      # kept for backward compat (single-point)
    self.multi_point_proxy = multi_point_proxy       # new: MultiPointProxyWrapper or None
    self._last_mp_aux_loss = 0.0                     # stored after forward
```

**Modified `forward`:**
```python
def forward(self, batch, proxy_embeddings=None, ...):
    dense_x, dense_mask = self.encode_dense(batch)
    B, max_N, d = dense_x.shape

    if self.multi_point_proxy is not None:
        # ── Multi-point mode ──
        insertion_set = set(self.multi_point_proxy.insertion_layers)
        point_idx = 0
        total_aux = 0.0
        aug_x = dense_x
        aug_mask = dense_mask

        for i, layer in enumerate(self.layers):
            if i in insertion_set:
                aug_x, aug_mask, aux = self.multi_point_proxy.run_proxy_block(
                    aug_x, aug_mask, point_idx
                )
                decay = self.multi_point_proxy.aux_loss_decay ** point_idx
                total_aux = total_aux + aux * decay
                point_idx += 1

            aug_x = layer(aug_x, aug_mask)

        self._last_mp_aux_loss = total_aux
        dense_x = aug_x
        dense_mask_for_readout = aug_mask

    elif proxy_embeddings is not None and self.cross_attn_router is not None:
        # ── Single-point cross-attention (existing behavior) ──
        dense_x = self.cross_attn_router(dense_x, proxy_embeddings, dense_mask)
        for layer in self.layers:
            dense_x = layer(dense_x, dense_mask)
        dense_mask_for_readout = dense_mask

    elif proxy_embeddings is not None:
        # ── Single-point N+M concat (existing behavior) ──
        M = proxy_embeddings.shape[1]
        dense_x = torch.cat([dense_x, proxy_embeddings], dim=1)
        aug_mask = torch.cat([
            dense_mask,
            torch.ones(B, M, dtype=torch.bool, device=dense_x.device)
        ], dim=1)
        for layer in self.layers:
            dense_x = layer(dense_x, aug_mask)
        dense_mask_for_readout = aug_mask

    else:
        # ── No proxies ──
        for layer in self.layers:
            dense_x = layer(dense_x, dense_mask)
        dense_mask_for_readout = dense_mask

    # Readout (unchanged logic, uses dense_mask_for_readout)
    ...
```

**Key detail for concat mode readout:** When multi-point concat accumulates N+kM tokens, the `readout_scope="nodes_only"` path uses `dense_x[:, :max_N, :]` which correctly extracts the first N positions (original nodes). The `"all_tokens"` path pools over everything including all accumulated proxies.

### 4.3 `models.py` — `GREDHybridTransformer`

Same pattern as GraphTransformer, but applied to `self.transformer_layers`. GRED layers run first as a block (unchanged), then the transformer layer loop gets the multi-point interleaving.

Note: the insertion layer indices are relative to the **transformer** layers, not the GRED layers. So `insertion_layers=[0]` means "before the first transformer layer" (after all GRED layers), which matches the current single-point position.

### 4.4 Main Training Scripts

All four scripts (`main_e2e.py`, `main_three_staged.py`, `main_indist.py`, `main_staged.py`) need:

**New CLI flags:**
```python
p.add_argument("--proxy_insertion_layers", type=str, default="-1",
               help="Comma-separated layer indices for multi-point proxy insertion. "
                    "-1 = no insertion (default). 0 = before layer 0 (like current). "
                    "E.g., '0,2,4' for three insertion points.")
p.add_argument("--separate_proxy_generators", action="store_true", default=False,
               help="Use independent generator per insertion point (Option B).")
p.add_argument("--separate_proxy_routers", action="store_true", default=False,
               help="Use independent router per insertion point.")
p.add_argument("--proxy_aux_loss_decay", type=float, default=1.0,
               help="Geometric decay factor for multi-point aux losses.")
```

**New helper function:**
```python
def _parse_insertion_layers(s):
    """Parse '--proxy_insertion_layers' string into list or None."""
    layers = [int(x.strip()) for x in s.split(",")]
    if layers == [-1]:
        return None  # disabled
    return sorted(layers)

def _build_multi_point_proxy(args, generator, router):
    """Build MultiPointProxyWrapper if multi-point is enabled."""
    layers = _parse_insertion_layers(args.proxy_insertion_layers)
    if layers is None:
        return None
    return MultiPointProxyWrapper(
        generator=generator,
        router=router,  # None if not using cross-attn routing
        insertion_layers=layers,
        separate_generators=args.separate_proxy_generators,
        separate_routers=args.separate_proxy_routers,
        aux_loss_decay=args.proxy_aux_loss_decay,
    )
```

**Modified `build_model()`:**
```python
def build_model(args):
    generator = _build_generator(args)
    cross_attn_router = _build_cross_attn_router(args)

    mp_wrapper = _build_multi_point_proxy(args, generator, cross_attn_router)

    if args.backbone == "vanilla_gt":
        model = GraphTransformer(
            ...,
            cross_attn_router=cross_attn_router if mp_wrapper is None else None,
            multi_point_proxy=mp_wrapper,
        )
    elif args.backbone == "hybrid":
        model = GREDHybridTransformer(
            ...,
            cross_attn_router=cross_attn_router if mp_wrapper is None else None,
            multi_point_proxy=mp_wrapper,
        )
    ...
    return model, generator  # generator still returned for non-multi-point paths
```

**Key:** When `multi_point_proxy` is active, we pass `cross_attn_router=None` to the model (the router lives inside the wrapper). When it's not active, we pass the router directly as before.

**Modified forward functions (e.g., `forward_e2e`):**
```python
def forward_e2e(batch, model, generator, ...):
    if model.multi_point_proxy is not None:
        # Multi-point: model handles generation + routing internally
        logits, node_emb = model(batch)
        aux_loss = model._last_mp_aux_loss
    else:
        # Existing flow: generate proxies externally, pass to model
        dense_x, dense_mask = model.encode_dense(batch)
        proxy_emb, aux_loss = generator(dense_x, dense_mask)
        logits, node_emb = model(batch, proxy_embeddings=proxy_emb,
                                  precomputed_dense=(dense_x, dense_mask))
    ...
```

### 4.5 Aux Loss Retrieval

Multi-point mode stores the accumulated aux_loss as `model._last_mp_aux_loss` (set during forward). This avoids changing the return signature of `model.forward()`, keeping full backward compatibility.


## 5. File Change Summary

| File | Changes |
|------|---------|
| `generators.py` | Add `import copy`. Add `MultiPointProxyWrapper` class. |
| `models.py` | Add `multi_point_proxy` param + `_last_mp_aux_loss` attribute to `GraphTransformer.__init__` and `GREDHybridTransformer.__init__`. Rewrite layer loop in both forward methods to support interleaved proxy blocks (multi-point branch + existing branches unchanged). |
| `main_e2e.py` | Add CLI flags. Add `_parse_insertion_layers()`, `_build_multi_point_proxy()`. Modify `build_model()` to wire wrapper. Modify `forward_e2e()` to read `_last_mp_aux_loss`. Import `MultiPointProxyWrapper`. |
| `main_three_staged.py` | Same pattern as main_e2e.py. |
| `main_indist.py` | Same pattern. Multi-point interacts with the 5-phase pipeline — only phases that call the full model benefit. |
| `main_staged.py` | Same pattern, using `_build_graph_transformer()` helper. |


## 6. Hyperparameter Recommendations

### For Peptides-func (6-layer GraphTransformer, hidden_dim=64, M=8 proxies)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `proxy_insertion_layers` | `0,2,4` | 3 insertion points, one per 2-layer block |
| `separate_proxy_generators` | `False` | Shared generator reduces overfitting |
| `separate_proxy_routers` | `False` | Weight sharing, Perceiver IO validated this |
| `proxy_aux_loss_decay` | `1.0` | No decay initially |
| `num_cross_layers` (router) | `1` | Lighter per-point routing since we have multiple points |

### Ablation Schedule

1. **Baseline:** Single-point at layer 0 only (`--proxy_insertion_layers 0`).
2. **Multi-point, shared generator:** `--proxy_insertion_layers 0,2,4`
3. **Multi-point, separate generators:** `--proxy_insertion_layers 0,2,4 --separate_proxy_generators`
4. **Multi-point, separate routers:** `--proxy_insertion_layers 0,2,4 --separate_proxy_routers`
5. **Varying insertion density:** `0,2,4` vs `0,3` vs `2` (single mid-point) vs `0,1,2,3,4,5` (every layer)
6. **Concat mode multi-point:** same insertion configs without `--use_cross_attn_routing` (tokens accumulate)
7. **Insertion + reduced router depth:** `--num_cross_layers 1` with multi-point vs `--num_cross_layers 2` single-point (same total compute budget)


## 7. Complexity & Parameter Analysis

### Current Single-Point (for reference)
- Generator: ~O(NM·d) per invocation
- Router (cross-attn): 2 cross-attention layers × (gather + self-refine + readback) = 6 attention ops
- Total: 6 × O(NM·d) per forward pass

### Multi-Point with K Insertion Points

**Option A — Shared Generator (default):**
- Generator runs K times: K × O(NM·d) FLOPs, but **zero additional parameters**
- Router runs K times: K × (num_cross_layers × 3) attention ops, **zero additional parameters** if shared
- Total param overhead: **0** — only FLOPs increase by factor K

**Option B — Separate Generators:**
- Generator params × K. For ScoreBasedGenerator with d=64, M=8: ~35K params per generator → 105K for K=3
- Router: same as above (shared or separate independently)

**Concat mode multi-point (K insertion points):**
- Layer 0 self-attention: O((N+M)² · d)
- Layer after 2nd insertion: O((N+2M)² · d)
- Layer after Kth insertion: O((N+KM)² · d)
- For N=100, M=8, K=3: final layer sees 124 tokens vs 108 for single-point. Modest increase.

### Comparison: Multi-point K=3 shared vs Single-point with 2 cross-layers

| | Single-point (current) | Multi-point K=3, shared, 1 cross-layer |
|---|---|---|
| Generator calls | 1 | 3 |
| Router attention ops | 6 (2 layers × 3 ops) | 9 (3 points × 1 layer × 3 ops) |
| Extra parameters | 0 | 0 |
| Information | Proxies see encoder output only | Proxies see encoder, layer-1, layer-3 outputs |


## 8. Implementation Order

1. Add `MultiPointProxyWrapper` to `generators.py`.
2. Add `multi_point_proxy` support to `GraphTransformer` (init + forward).
3. Add `multi_point_proxy` support to `GREDHybridTransformer` (init + forward).
4. Add CLI flags and wiring in `main_e2e.py`.
5. Propagate to `main_three_staged.py`, `main_indist.py`, `main_staged.py`.
6. Verify backward compatibility: `--proxy_insertion_layers -1` = no proxies, `--proxy_insertion_layers 0` = identical to current single-point behavior.
7. Run ablation experiments.


## 9. Open Design Questions

1. **Should `--proxy_insertion_layers 0` with multi-point wrapper produce identical results to the current single-point code path?** Yes — this is a hard requirement for backward compatibility verification. The wrapper with `insertion_layers=[0]` and shared generator/router should be numerically equivalent.

2. **Training stability with multiple generator calls.** Multiple aux losses per forward pass mean more gradient sources. Consider gradient clipping on aux losses or starting with `--proxy_aux_loss_decay 0.5` if training is unstable.

3. **Interaction between concat-mode accumulation and readout.** With `readout_scope="nodes_only"`, readout pools `dense_x[:, :max_N, :]` — the first N positions. In concat mode with multi-point, these first N positions are the original nodes that have been attending to accumulated proxies through self-attention. This is correct and desirable. With `readout_scope="all_tokens"`, all N+KM tokens are pooled, including all K batches of proxies.

4. **Validation of insertion layer indices.** The wrapper should validate that all insertion layer indices are in `[0, num_layers)` at construction time and raise a clear error otherwise.
