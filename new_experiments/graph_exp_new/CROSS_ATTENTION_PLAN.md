# Cross-Attention Routing Plan: N→M→N Hyperedge Proxies

## Motivation

Currently, all pipelines generate M proxy embeddings and **concatenate** them with N node embeddings to form N+M tokens for the transformer's self-attention. Every node attends to every proxy and every other node — O((N+M)²) cost with no structural bias.

The alternative, proposed across multiple documents (FLOW_MATCHING_PROPOSAL §2.2, END_TO_END_PROPOSAL §5.1 ablation), is **routed cross-attention**: use the M generated proxies as hyperedges in a bipartite attention scheme. Instead of outputting M proxies that join self-attention, we perform N→M→N cross-attention and output **N refined node embeddings** that have been informed by the M proxy hyperedges. This reduces attention cost to O(NM) and introduces an information bottleneck that forces proxies to specialize.

**Key idea:** The generator still produces M proxy embeddings. A new `CrossAttentionRouter` module then:
1. Lets M proxies **gather** information from N nodes (N→M cross-attention)
2. Lets M proxies **refine** among themselves (M×M self-attention)
3. Lets N nodes **read back** from M proxies (M→N cross-attention)

The output is N embeddings of the same dimensionality — these replace or augment the original node embeddings before the transformer layers. The M proxies never enter the transformer; they serve only as routing intermediaries.

---

## Architecture: `CrossAttentionRouter`

```
Input:
    node_embeddings: (B, N, d)   — from encoder or GRED
    proxy_embeddings: (B, M, d)  — from any generator
    node_mask: (B, N)            — True = real node

Output:
    refined_nodes: (B, N, d)     — proxy-informed node embeddings

Architecture (per layer, stacked L_cross times):

    Step 1 — Proxy Gather (N→M):
        Q = proxy_embeddings   (B, M, d)
        K, V = node_embeddings (B, N, d)
        proxy_updated = CrossAttn(Q=proxy, K=node, V=node) + proxy   # residual
        proxy_updated = LayerNorm + FFN + residual

    Step 2 — Proxy Self-Refine (M×M):
        proxy_refined = SelfAttn(proxy_updated) + proxy_updated       # residual
        proxy_refined = LayerNorm + FFN + residual

    Step 3 — Node Readback (M→N):
        Q = node_embeddings    (B, N, d)
        K, V = proxy_refined   (B, M, d)
        node_updated = CrossAttn(Q=node, K=proxy, V=proxy) + node    # residual
        node_updated = LayerNorm + FFN + residual

    → node_embeddings = node_updated  (for next layer iteration)
    → proxy_embeddings = proxy_refined (for next layer iteration)

Final output: node_updated (B, N, d)
```

### Design Choices

- **Residual connections** at every step — the refined nodes always retain original information.
- **Multiple layers** (`num_cross_layers`, default 2) allow iterative refinement. Each layer re-routes through the same M proxies.
- **Both proxy and node states evolve** across layers — proxies accumulate increasingly abstract summaries; nodes receive increasingly refined global context.
- **Key padding mask** applied in N→M cross-attention to ignore padding nodes.
- **No padding mask needed** for M→N since all M proxies are valid.

---

## Module Definition (to be added to `generators.py`)

```python
class CrossAttentionRouter(nn.Module):
    """
    N→M→N cross-attention routing using M proxies as hyperedges.

    Takes M proxy embeddings from any generator and N node embeddings,
    performs bidirectional cross-attention routing, and returns N refined
    node embeddings. The M proxies never enter the transformer.

    Args:
        hidden_dim: embedding dimension d.
        num_heads: attention heads for cross- and self-attention.
        num_cross_layers: number of N→M→N routing iterations.
        dropout: dropout rate.
        use_proxy_self_attn: if True, include M×M self-attention in each layer.
    """
    def __init__(self, hidden_dim, num_heads=8, num_cross_layers=2,
                 dropout=0.2, use_proxy_self_attn=True):
        ...

    def forward(self, node_embeddings, proxy_embeddings, node_mask):
        """
        Args:
            node_embeddings: (B, N, d)
            proxy_embeddings: (B, M, d)
            node_mask: (B, N) boolean — True = real node
        Returns:
            refined_nodes: (B, N, d)
        """
        ...
```

---

## Integration Points: Required Code Changes

### 1. `generators.py` — Add `CrossAttentionRouter` class

**New class** at module level (not inside any generator). This keeps it generator-agnostic — any generator's M-dimensional output can be routed.

The class is standalone, ~80 lines. Each layer has:
- 2 `nn.MultiheadAttention` (cross-attn N→M and M→N)
- 1 `nn.MultiheadAttention` (self-attn M×M, conditional on `use_proxy_self_attn`)
- 3 `nn.LayerNorm` + 3 FFN blocks (one per attention sublayer)

**Parameters to add to `BaseGenerator`:** None. The router is a separate module, not part of the generator. It sits between the generator and the model.

### 2. `models.py` — New forward path in `GraphTransformer` and `GREDHybridTransformer`

#### `GraphTransformer.forward()`

Current proxy path (lines 177-184):
```python
if proxy_embeddings is not None:
    M = proxy_embeddings.shape[1]
    dense_x = torch.cat([dense_x, proxy_embeddings], dim=1)  # N+M concat
    aug_mask = torch.cat([dense_mask, ones_M], dim=1)
```

**New path** when `cross_attn_router` is provided:
```python
if proxy_embeddings is not None and self.cross_attn_router is not None:
    # N→M→N routing: produces (B, N, d), no concat
    dense_x = self.cross_attn_router(dense_x, proxy_embeddings, dense_mask)
    aug_mask = dense_mask  # still N tokens, not N+M
elif proxy_embeddings is not None:
    # Original concat path
    dense_x = torch.cat([dense_x, proxy_embeddings], dim=1)
    aug_mask = torch.cat([dense_mask, ones_M], dim=1)
```

**Constructor change** — add optional `cross_attn_router`:
```python
class GraphTransformer(nn.Module):
    def __init__(self, ..., cross_attn_router=None):
        ...
        self.cross_attn_router = cross_attn_router  # None = use concat (default)
```

The router is passed in from outside (constructed in `build_model()` or the training script), not built internally. This keeps the model class clean and the routing decision external.

#### `GREDHybridTransformer.forward()`

Same pattern — in the proxy integration section (lines 657-665), add the router branch before the concat fallback. The router receives GRED-encoded node features `h` and proxy embeddings, returns refined `h`.

```python
if proxy_embeddings is not None and self.cross_attn_router is not None:
    h = self.cross_attn_router(h, proxy_embeddings, dense_mask)
    h_aug = h
    aug_mask = dense_mask
elif proxy_embeddings is not None:
    # existing concat path
    h_aug = torch.cat([h, proxy_embeddings], dim=1)
    aug_mask = torch.cat([dense_mask, ones_M], dim=1)
```

**Constructor change** — same as `GraphTransformer`:
```python
class GREDHybridTransformer(nn.Module):
    def __init__(self, ..., cross_attn_router=None):
        ...
        self.cross_attn_router = cross_attn_router
```

#### Impact on readout

With the router, there are always N tokens (not N+M), so the `readout_scope` logic simplifies — `"all_tokens"` and `"nodes_only"` become identical. No special handling needed. The existing `nodes_only` path works unchanged.

### 3. `main_e2e.py` — CLI flag + forward pass changes

#### New CLI arguments:
```python
p.add_argument("--use_cross_attn_routing", action="store_true", default=False,
               help="Use N→M→N cross-attention routing instead of N+M concat")
p.add_argument("--num_cross_layers", type=int, default=2,
               help="Number of N→M→N routing layers (only with --use_cross_attn_routing)")
p.add_argument("--cross_attn_proxy_self_attn", action=argparse.BooleanOptionalAction,
               default=True, help="Include M×M self-attention within cross-attention routing")
```

#### `build_model()` change:
```python
def build_model(args):
    # Build router if requested
    cross_attn_router = None
    if getattr(args, 'use_cross_attn_routing', False):
        from generators import CrossAttentionRouter
        cross_attn_router = CrossAttentionRouter(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_cross_layers=args.num_cross_layers,
            dropout=args.dropout,
            use_proxy_self_attn=args.cross_attn_proxy_self_attn,
        )

    if args.backbone == "vanilla_gt":
        return GraphTransformer(..., cross_attn_router=cross_attn_router)
    elif args.backbone == "hybrid":
        return GREDHybridTransformer(..., cross_attn_router=cross_attn_router)
```

**Important:** The router's parameters must be included in the optimizer. Since the router is a submodule of the model (`self.cross_attn_router`), `model.parameters()` will automatically include it.

#### `forward_e2e()` — No changes needed

The magic is that `forward_e2e()` already calls `model.forward(..., proxy_embeddings=proxy_emb)`. The model internally decides whether to concat or route based on `self.cross_attn_router`. The forward function doesn't need to know which path is taken.

### 4. `main_three_staged.py` — Same CLI flags + same `build_model()` change

The three-staged pipeline uses the same model construction pattern. The changes mirror `main_e2e.py`:
- Add `--use_cross_attn_routing`, `--num_cross_layers`, `--cross_attn_proxy_self_attn` to argparse.
- Modify `build_model()` to pass `cross_attn_router` to the backbone constructor.
- The stage 2 (generator training through frozen transformer) and stage 3 (fine-tuning) both call `model.forward()` with `proxy_embeddings=`, so the routing happens transparently.

### 5. `main_indist.py` — Same pattern

The in-distribution pipeline's Phase 3 (generator training) and Phase 5 (fine-tuning) both use `model.forward()` with `proxy_embeddings=`. Same CLI flags, same `build_model()` modification.

### 6. `main_staged.py` — Same pattern

Stage 3 (flow matching / score-based generator training) and Stage 4 (fine-tuning) both pass proxy_embeddings to the model. Same changes apply.

---

## Summary of Files Changed

| File | Change Type | Description |
|------|------------|-------------|
| `generators.py` | **New class** | Add `CrossAttentionRouter` (~80-100 lines) |
| `models.py` | **Modify** `GraphTransformer.__init__` | Add optional `cross_attn_router` param |
| `models.py` | **Modify** `GraphTransformer.forward` | Add router branch before concat |
| `models.py` | **Modify** `GREDHybridTransformer.__init__` | Add optional `cross_attn_router` param |
| `models.py` | **Modify** `GREDHybridTransformer.forward` | Add router branch before concat |
| `main_e2e.py` | **Modify** argparse + `build_model()` | Add 3 CLI flags, pass router to model |
| `main_three_staged.py` | **Modify** argparse + `build_model()` | Same 3 CLI flags, same `build_model()` change |
| `main_indist.py` | **Modify** argparse + `build_model()` | Same pattern |
| `main_staged.py` | **Modify** argparse + `build_model()` | Same pattern |

---

## What Does NOT Change

- **Generator classes** (`ScoreBasedGenerator`, `PMAGenerator`, `GraphCoarseningGenerator`, `GNNPoolingGenerator`, `FlowMatchingGenerator`): Completely unchanged. They still output `(B, M, d)` proxy embeddings. The router is downstream of them.
- **`data.py`**: No data changes needed.
- **`metrics.py`**: No changes.
- **`mmd.py`**: No changes.
- **`GREDEncoder`** (standalone GRED without transformer layers): Already doesn't support proxies; remains unchanged.
- **Readout logic**: With routing, there are N tokens not N+M, so the existing `nodes_only` readout path works as-is.

---

## New Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_cross_attn_routing` | `False` | Enable N→M→N routing (flag) |
| `--num_cross_layers` | `2` | Number of routing iterations |
| `--cross_attn_proxy_self_attn` | `True` | Include M×M self-attention in routing |

These are added to all four main scripts. When `--use_cross_attn_routing` is not set, behavior is identical to the current concat approach — fully backward compatible.

---

## Ablation Plan

When implementing, the following ablations should be run:

1. **Concat vs. routing**: Same generator, same backbone, same hyperparameters — only toggle `--use_cross_attn_routing`. This is the primary comparison.
2. **Number of routing layers**: `--num_cross_layers` ∈ {1, 2, 3, 4}. More layers = more refinement but also more parameters and compute.
3. **Proxy self-attention**: `--cross_attn_proxy_self_attn` on vs. off. Tests whether M×M self-attention adds value or whether the two cross-attention steps suffice.
4. **M sensitivity under routing**: `--num_proxies` ∈ {4, 8, 16, 32, 64}. The bottleneck effect should make routing more sensitive to M than concat.
5. **Routing + readout scope**: With routing, `all_tokens` and `nodes_only` are equivalent (only N tokens exist). Verify this doesn't regress.

---

## Interaction with Existing Pipelines

### End-to-End (`main_e2e.py`)
- Generator + router + transformer all trained jointly from scratch.
- Router parameters are part of the model and get gradients from task loss.
- Proxy warmup (`--proxy_warmup_epochs`) works as before: during warmup, no proxies are generated, no routing happens.

### Three-Staged (`main_three_staged.py`)
- **Stage 1**: Pretrain transformer (no proxies, no router). Router is part of model but receives no proxy input, so it's a no-op.
- **Stage 2**: Train generator through frozen transformer+router. The router is frozen along with the transformer. Gradients flow: task_loss → frozen transformer → frozen router → generator.
- **Stage 3**: Unfreeze all. Router, generator, and transformer all fine-tune together.

### In-Distribution (`main_indist.py`)
- **Phase 1**: Full transformer baseline (no proxies, router is no-op).
- **Phase 2**: Partial-graph training (no proxies, router is no-op).
- **Phase 3**: Generator training with frozen model+router.
- **Phase 4**: Augmented evaluation — router processes generated proxies.
- **Phase 5**: End-to-end fine-tuning with router.

### Staged with Optimization Targets (`main_staged.py`)
- **Stage 2** (proxy optimization): The proxy optimization step optimizes free proxy embeddings against the frozen transformer. With routing, the optimized proxies are routed through the frozen `CrossAttentionRouter` before entering the transformer. The optimization target shape is still `(M, d)` — unchanged.
- **Stage 3**: Generator trained to produce M proxies that, when routed, improve the task. Works identically.
- **Stage 4**: Fine-tune all components.

---

## Implementation Order

1. **Add `CrossAttentionRouter` to `generators.py`** — self-contained, no dependencies on other changes.
2. **Modify `GraphTransformer` and `GREDHybridTransformer` in `models.py`** — add `cross_attn_router` parameter and the routing branch in `forward()`.
3. **Modify `main_e2e.py`** — add CLI flags and update `build_model()`. Test with `--use_cross_attn_routing`.
4. **Propagate to other main scripts** — `main_three_staged.py`, `main_indist.py`, `main_staged.py` — same mechanical changes.
5. **Run ablations** — concat vs. routing, varying M and routing depth.

---

## Parameter Count Impact

The `CrossAttentionRouter` with `num_cross_layers=2`, `hidden_dim=256`, `num_heads=8` adds:

Per layer:
- 2 cross-attention: 2 × (3 × 256² + 256²) = 2 × 4 × 65536 = 524,288 params
- 1 self-attention: 4 × 65536 = 262,144 params
- 3 FFN (d→4d→d each): 3 × (256×1024 + 1024×256) = 3 × 524,288 = 1,572,864 params
- 6 LayerNorm: 6 × 512 = 3,072 params

Per layer total: ~2.36M params
2 layers total: **~4.72M params**

For comparison, the existing 6-layer transformer with hidden_dim=256 has ~6 × (4 × 256² + 2 × 256 × 1024) = ~9.4M params. The router adds ~50% overhead but replaces the N+M token expansion in the transformer, which had O((N+M)²) attention cost per layer.

---

## Complexity Analysis

| Approach | Attention cost per transformer layer | Total attention layers |
|----------|-------------------------------------|----------------------|
| Concat (current) | O((N+M)² × d) | L_transformer |
| Routing (proposed) | O(NM × d) in router + O(N² × d) in transformer | L_cross + L_transformer |

For N=200, M=32, L_transformer=6, L_cross=2:
- Concat: 6 × (200+32)² = 6 × 53,824 = 322,944 attention pairs
- Routing: 2 × (200×32 + 32² + 200×32) + 6 × 200² = 2 × 13,824 + 240,000 = 267,648 attention pairs

Routing is **~17% cheaper** for these typical Peptides-func dimensions, and the savings increase with larger M.
