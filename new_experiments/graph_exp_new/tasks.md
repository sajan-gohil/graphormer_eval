# Implementation Plan: Proxy Node Graph Transformer

## Design Overview

Two pipelines share a common transformer backbone and swappable proxy generators:

**Pipeline A — Staged (with proxy optimization targets):**
```
Stage 1: Pretrain GraphTransformer → freeze
Stage 2: Per-graph proxy optimization → save target embeddings
Stage 3: Train a generator to predict those targets from node embeddings
Stage 4: End-to-end finetune (generator + transformer + embeddings)
```

**Pipeline B — End-to-End (no optimization stage):**
```
Single stage: Train generator + transformer jointly from scratch using only task loss
```

Both pipelines support swappable generator architectures:
1. **FlowMatchingGenerator** — conditional denoiser with cross-attention (Pipeline A only)
2. **ScoreBasedGenerator** — node scoring + softmax aggregation + self-attention refinement
3. **GNNPoolingGenerator** — multi-hop GNN + multi-scale pooling + shared MLP decode

All generators share the same interface:
- Input: `node_embeddings (B, N, d)`, `mask (B, N)`, optional `targets (B, M, d)`, optional graph-structure kwargs
- Output: `proxy_embeddings (B, M, d)`, `aux_loss` (reconstruction/FM loss when targets given, else None)

---

## File Structure (7 files)

```
models.py              — NodeEncoder, TransformerLayer, GraphTransformer
generators.py          — BaseGenerator, FlowMatchingGenerator, ScoreBasedGenerator, GNNPoolingGenerator
mmd.py                 — Already exists, keep as-is
metrics.py             — Already exists, minor update to collect_predictions
data.py                — get_loaders(), ProxyTargetDataset, collate_with_proxies
main_staged.py         — CLI + training loop for Pipeline A (stages 1-4)
main_e2e.py            — CLI + training loop for Pipeline B
```

---

## Module Specifications

### 1. `models.py` — Transformer Backbone

Rewrite or reuse from `vanilla_gt.py` and `optimization_phase_sample.py` with proper dropout, masking, and proxy insertion support.

#### `NodeEncoder(hidden_dim)`
- Reuse from `optimization_phase_sample.py` (AtomEncoder + BondEncoder)
- Input: `x, edge_index, edge_attr` (flat pyg tensors)
- Output: `(total_N, d)` node embeddings

#### `TransformerLayer(hidden_dim, num_heads, dropout)`
- Pre-norm architecture (LayerNorm before attention and FFN)
- Q/K/V projections → multi-head scaled dot-product attention → output projection
- FFN: Linear(d, 4d) → GELU → Dropout → Linear(4d, d)
- Dropout on: attention weights, FFN intermediate, residual paths
- Attention masking: accept `mask (B, N)`, apply to both key and query positions (pad positions get -inf). Same logic as `optimization_phase_sample.py:TransformerLayer.forward`
- Input: `x (B, N, d)`, `mask (B, N)` optional
- Output: `x (B, N, d)`

#### `GraphTransformer(num_layers, num_heads, hidden_dim, output_dim, dropout, ...)`
- Contains: `NodeEncoder`, `ModuleList[TransformerLayer]`, classification head (Linear → ReLU → Dropout → Linear)
- `encode_nodes(batch)` → flat `(total_N, d)` embeddings
- `encode_dense(batch)` → `(B, max_N, d)`, `(B, max_N)` mask via `to_dense_batch`
- `forward(batch, proxy_embeddings=None, precomputed_dense=None)`:
  - If `precomputed_dense` provided, skip encoding (used in E2E pipeline to avoid double encoding)
  - If `proxy_embeddings (B, M, d)` provided, concatenate with dense_x along dim=1, extend mask with True for proxy positions
  - Run through transformer layers with combined mask
  - Extract original node positions `dense_x[:, :max_N, :]`, apply original mask, global_mean_pool, classification head
  - Return `logits (B, output_dim)`, `node_embeddings (total_N, d)`
- Follow pattern from `optimization_phase_sample.py:GraphTransformer` but with configurable dropout

### 2. `generators.py` — Proxy Generator Architectures

All generators inherit from a base and implement a common interface. **No normalization or activation on the final output** (targets from optimization are unconstrained).

#### `BaseGenerator` (abstract)
- Stores `num_proxies (M)`, `hidden_dim (d)`
- Abstract method: `forward(node_embeddings, mask, targets=None, **kwargs) → (proxy_embeddings, aux_loss)`
- Concrete method: `generate(node_embeddings, mask, **kwargs) → proxy_embeddings` — calls forward with targets=None, returns just the proxies

#### `FlowMatchingGenerator(num_proxies, node_dim, denoiser_dim, denoiser_layers, denoiser_heads, dropout, euler_steps)`
- **Only used in Pipeline A** (needs target embeddings for training)
- **Denoiser architecture**: stack of L layers, each:
  - Additive time conditioning: `x_t = x_t + time_emb`
  - Cross-attention: `Q=x_t, K=node_embeddings, V=node_embeddings` (with node mask applied to key positions)
  - FFN with residual + LayerNorm
- Time embedding: sinusoidal encoding → MLP(sin_dim → denoiser_dim)
- If `node_dim != denoiser_dim`: add input projection for node embeddings and output projection back to node_dim
- `forward(node_embeddings, mask, targets=None, **kwargs)`:
  - **Training (targets provided)**: sample `t ~ U(0,1)`, `x_0 ~ N(0,I) shape (B,M,d)`, interpolate `x_t = (1-t)*x_0 + t*targets`, target vector field `u = targets - x_0`, predict `v = denoiser(x_t, t, node_embeddings)`, loss = `MSE(v, u)`. Return (generated_proxies_via_euler, loss)
  - **Inference (no targets)**: sample `z_0 ~ N(0,I)`, run `euler_steps` Euler integration steps from t=0 to t=1. Return (B, loss=None)
- Time `t` is per-sample in the batch (shape `(B,)`)

#### `ScoreBasedGenerator(num_proxies, input_dim, hidden_dim, num_layers, num_heads, dropout)`
- `input_dim` = node embedding dim (matches transformer hidden_dim)
- Step 1 — Value projection: `V = Linear(X)` → (B, N, d)
- Step 2 — Score MLP: `S = MLP(X)` → (B, N, M). MLP: Linear(d, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, M)
- Step 3 — Masked softmax: set padding positions in S to -inf using mask, then softmax over node dim (dim=1) → attention weights `A (B, N, M)`
- Step 4 — Aggregate: `B0 = A.transpose(1,2) @ V` → (B, M, d)
- Step 5–6 — L refinement layers, each: self-attention among M proxies (standard multi-head) + FFN, with residual + LayerNorm
- No final activation/normalization on output
- `forward(node_embeddings, mask, targets=None, **kwargs)`:
  - Run steps 1–6 → proxy_embeddings (B, M, d)
  - If targets: `aux_loss = MMD(proxy_embeddings, targets)` (per-graph, averaged over batch) or MSE. Use MMD from `mmd.py`
  - Return (proxy_embeddings, aux_loss)

#### `GNNPoolingGenerator(num_proxies, input_dim, gnn_layers, gnn_type, pool_types, decode_hidden, decode_layers, idx_emb_dim, dropout, decode_mode)`
- Needs `edge_index` and optionally `edge_attr` — passed via `**kwargs` along with `batch_vec` (pyg batch vector for proper pooling)
- Step 1 — Multi-hop GNN: K GNN layers (GCN/GIN/GINE/GAT configurable), each with residual + LayerNorm. Collect H^0...H^K (K+1 representations). Operate on **flat** pyg tensors (not dense batch) since message passing needs edge_index.
- Step 2 — Multi-scale pooling: for each hop level, apply pool functions (mean/max/std, configurable subset) using `batch_vec` to respect graph boundaries. Each pool → (B_s, d). Concatenate all → descriptor `g (B_s, D)` where `D = P*(K+1)*d`
- Step 3 — Decode M proxies via shared MLP:
  - Proxy index embeddings: `E (M, idx_emb_dim)` learnable
  - For each graph: `input = [g_repeated; E]` → (M, D+idx_emb_dim), MLP → (M, d)
  - MLP: Linear(D+idx_emb_dim, decode_hidden) → GELU → Dropout → Linear(decode_hidden, decode_hidden) → GELU → Dropout → Linear(decode_hidden, d)
  - No final activation
- Alternative `decode_mode='grouped'`: split M into G groups, each group has own MLP outputting (M/G)*d
- `forward(node_embeddings, mask, targets=None, **kwargs)`:
  - Expects `kwargs['edge_index']`, `kwargs['batch_vec']`, optionally `kwargs['edge_attr']`
  - `node_embeddings` here is flat `(total_N, d)` — GNN needs flat format. Convert to dense `(B, M, d)` at the end.
  - If targets: aux_loss = MMD
  - Return (proxy_embeddings, aux_loss)

### 3. `mmd.py` — Keep existing
- `median_heuristic`, `gaussian_kernel`, `mmd_squared` — no changes needed

### 4. `metrics.py` — Minor updates
- Keep `compute_macro_ap`, `compute_per_class_ap`, `compute_correct_classes`
- Update `collect_predictions` to handle the model returning `(logits, node_emb)` tuple — currently calls `model(batch)` expecting just logits. Fix to unpack.
- Add a batched AP evaluation helper that takes model + loader + optional proxy_fn and returns (AP, loss). Essentially the `evaluate()` function from `optimization_phase_sample.py`.

### 5. `data.py` — Data Loading

#### `get_loaders(batch_size, num_workers=4)`
- Load LRGBDataset Peptides-func train/val/test
- Return (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
- Same pattern as `optimization_phase_sample.py:get_loaders`

#### `ProxyTargetDataset(pyg_dataset, proxy_pairs_path)`
- Load saved proxy pairs from pickle
- For each graph, keep the best proxy set (lowest opt_loss across repeats) from one optimization cycle, but keep results of all optimization cycles. Follow batching done in `optimization_phase_sample.py:phase2_optimize_proxies`
- Graphs without a valid proxy are not included in the proxy list, but if after completing `num_steps` number of optimization iterations, try for `num_steps` more iterations for these graphs. Else, save best variant (that has minimum task loss above threshold and lowest mmd loss associated with it)
- The proxies should be mapped to the graph it came from. Original samples are not be needed as is, so new samples with the optimized proxies can be saved as train data for rest of the stages of  pipeline A.


### 6. `main_staged.py` — Pipeline A Orchestration

CLI args (argparse) with yaml config override. CLI always wins over yaml. Key args:
- `--stage {1,2,3,4,all}` — which stage to run (default: all)
- `--generator {flow_matching,score_based,gnn_pooling}` — which generator to use in stage 3/4
- `--config path/to/config.yaml` — optional yaml
- All hyperparameters from the proposals as CLI args with defaults from Section 6.2

#### Stage 1 — Pretrain Transformer
- Function: `run_stage1(args) → model_path`
- Train GraphTransformer with BCE loss, AdamW, gradient clipping, early stopping on val AP (patience from args)
- Log every epoch: train_loss, train_AP, val_loss, val_AP. Log test_AP every 10 epochs.
- Save best checkpoint by val AP
- Return path to best checkpoint

#### Stage 2 — Optimize Proxy Embeddings
- Function: `run_stage2(args, model_path) → proxy_pairs_path`
- Load frozen transformer from stage 1
- Batch-aware optimization: process train_loader in batches (not single graphs). For each graph in batch, maintain independent proxy parameters `(1, M, d)` initialized with `randn * 0.02`
- Optimization loop per batch: forward with proxies → task_loss + mmd_lambda * MMD(proxies, node_embeddings) → backward only on proxy params
- Track per-graph: base_loss (without proxies), best_loss, base_correct_classes, best_correct_classes
- Save criteria: loss < threshold OR at least one new correct class
- Run `num_restarts` times (different random inits), keep all valid pairs
- Compute MMD distribution statistics across all original node embeddings first. After optimization, discard samples where proxy MMD > 3 std from mean.
- Log AP every 1000 samples (batch those 1000 sample results)
- Save pairs to pickle: list of dicts with `{encoder_emb, mask, proxy_emb, base_loss, opt_loss, sample_idx, original_sample, ...}` as needed.

#### Stage 3 — Train Generator
- Function: `run_stage3(args, model_path, proxy_pairs_path) → generator_path`
- Load frozen transformer, load proxy targets into ProxyTargetDataset
- Instantiate chosen generator
- Training loop: generator produces proxies from node embeddings, compute generator loss (FM loss or MMD against targets)
- **Downstream evaluation** every `eval_every` epochs (default 1):
  - Generate proxies for val set → insert into frozen transformer → compute val AP
  - Same for test set
  - This is the **primary checkpoint metric**, not reconstruction loss
- Log: generator_train_loss, generator_val_loss, downstream_val_AP, downstream_test_AP
- Early stop on downstream val AP
- Save diagnostic: collect (reconstruction_loss, downstream_AP) pairs every eval for correlation plot

#### Stage 4 — End-to-End Finetune
- Function: `run_stage4(args, model_path, generator_path) → final_model_path`
- Load transformer from stage 1, generator from stage 3
- Unfreeze everything (embedding + generator + transformer)
- For flow matching: use single Euler step (differentiable) during training
- Different LR groups: transformer params (low LR), generator params (higher LR), embedding (low LR)
- Train on task loss only (no proxy matching loss)
- Standard eval: val AP, test AP every epoch
- Early stop on val AP
- Save best checkpoint

#### Main block
- Parse args, run requested stages sequentially, passing paths between stages


### 7. `main_e2e.py` — Pipeline B Orchestration

CLI args similar to main_staged.py but simpler (no stage selection). Key args:
- `--generator {score_based,gnn_pooling}` — flow matching not supported here (no targets)
- `--mmd_lambda` — optional MMD regularization (0 to disable)
- `--proxy_warmup_epochs` — epochs to train transformer without proxies before activating generator
- `--readout_scope {nodes_only,all_tokens}` — whether to pool over N or N+M
- All model/training hyperparameters

#### Training loop
- Function: `run_e2e(args)`
- Build full model: NodeEncoder + chosen Generator + GraphTransformer
- Forward pass: encode nodes → generate proxies → concat → transformer → readout → classify
- The transformer's `forward` accepts `precomputed_dense` to avoid double encoding since the generator already ran the encoder
- Loss: `task_loss + mmd_lambda * MMD(proxies, node_embeddings)` (MMD optional, calculated always but set to 0 to ignore.)
- If `proxy_warmup_epochs > 0`: train transformer alone for that many epochs first (pass no proxies), then activate generator
- Single optimizer for everything
- Eval every epoch: val AP, test AP
- Early stop on val AP (patience 30)
- Log: train_loss, mmd_loss, val_AP, test_AP

---

## Implementation Order

### Phase 1: Core infrastructure
- [ ] 1.1 `data.py` — `get_loaders()`, `ProxyTargetDataset`, `collate_proxy_targets`
- [ ] 1.2 `models.py` — `NodeEncoder`, `TransformerLayer`, `GraphTransformer` with proxy insertion, configurable dropout
- [ ] 1.3 `metrics.py` — fix `collect_predictions` to handle (logits, emb) tuple, add `evaluate(model, loader, proxy_fn, device)` helper
- [ ] 1.4 Verify: load data, instantiate model, run one forward pass, confirm shapes

### Phase 2: Generators
- [ ] 2.1 `generators.py` — `BaseGenerator` with interface
- [ ] 2.2 `ScoreBasedGenerator` — simplest, implement first. Value projection, score MLP, masked softmax, aggregation, refinement layers
- [ ] 2.3 `FlowMatchingGenerator` — denoiser with cross-attention, time embedding, Euler sampling
- [ ] 2.4 `GNNPoolingGenerator` — GNN layers (support GCN/GIN/GINE/GAT), multi-pool, shared MLP decode with index embeddings
- [ ] 2.5 Verify: instantiate each generator, run forward with dummy data, confirm output shapes (B, M, d)

### Phase 3: Pipeline B (end-to-end)
- [ ] 3.1 `main_e2e.py` — argparse setup with yaml config override
- [ ] 3.2 Training loop: forward pass composition (encode → generate → transformer), loss, eval, early stopping, checkpoint saving
- [ ] 3.3 Proxy warmup support (train without proxies for K epochs)
- [ ] 3.4 Readout scope ablation (pool over N vs N+M)
- [ ] 3.5 Test run: end-to-end training with ScoreBasedGenerator for ~5 epochs, verify AP is computed and logged

### Phase 4: Pipeline A (staged)
- [ ] 4.1 `main_staged.py` — argparse setup, stage selection logic
- [ ] 4.2 Stage 1: pretrain transformer (train/eval loop, early stopping, checkpoint save)
- [ ] 4.3 Stage 2: proxy optimization (batched optimization, MMD constraint, filtering, save pairs). Adapt from `optimization_phase_sample.py` but make it batched and add MMD + filtering criteria
- [ ] 4.4 Stage 3: train generator with proxy targets, downstream eval on frozen transformer for val AP
- [ ] 4.5 Stage 4: end-to-end finetune with differential LR groups
- [ ] 4.6 Test run: full pipeline with ScoreBasedGenerator, verify stages chain correctly

### Phase 5: Polish
- [ ] 5.1 Diagnostic logging: collect (loss, AP) pairs for correlation analysis
- [ ] 5.2 Verify FlowMatchingGenerator in staged pipeline
- [ ] 5.3 Verify GNNPoolingGenerator in both pipelines
- [ ] 5.4 Parameter count / timing logging per epoch

---

## Key Design Decisions

1. **Generator interface is uniform.** All generators take `(node_embeddings, mask, targets, **kwargs)` and return `(proxies, aux_loss)`. Pipeline code never branches on generator type except for GNN needing edge_index in kwargs.

2. **GraphTransformer accepts `precomputed_dense`.** In end-to-end mode, the encoder runs once, output goes to both the generator and the transformer. Avoids double encoding.

3. **No final activation/norm on generator output.** Optimized proxy targets are unconstrained — generators must not clip or normalize their output.

4. **FlowMatchingGenerator is Pipeline A only.** It requires target proxy embeddings for the FM training objective. The other two generators work in both pipelines (with targets → compute aux_loss, without → just generate).

5. **Batched proxy optimization in Stage 2.** Process DataLoader batches, but each graph has its own proxy parameters. Use a list of per-graph optimizers or a single `(B, M, d)` parameter. The latter is simpler — initialize per batch, optimize for `num_steps`, track best per-graph.

6. **Single `generators.py` file** for all architectures since they share the base class and are not individually large.

7. **mmd.py stays separate** since it's a standalone utility already working.

8. **Config precedence**: CLI args > yaml file > hardcoded defaults. Use `argparse` defaults as the hardcoded defaults. If `--config` is provided, load yaml and set as defaults before parsing CLI args.
