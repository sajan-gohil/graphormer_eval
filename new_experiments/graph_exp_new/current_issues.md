# Current Issues

## 1. ip_mean / ip_std Printing Redundancy

Diversity loss and ip_mean (mean inter-proxy cosine similarity) convey the same information. Pipelines should print `ip_std` (standard deviation of inter-proxy cosine similarities) instead, as it provides additional insight into proxy distribution uniformity.

### main_adversarial_distillation.py

- **Stage 2 (~line 959):** Prints `div` (diversity loss) and `ip_mean`, but does **not** print `ip_std`. The `ip_std` is computed but unused.
  ```python
  # Currently:
  print(f"... div={diversity_val:.4f} ip_mean={ip_mean:.4f}")
  # Should be:
  print(f"... div={diversity_val:.4f} ip_std={ip_std:.4f}")
  ```

- **Stage 3 (~line 1165):** Computes `ip_mean` and `ip_std` via `inter_proxy_stats()` but **never prints either**. The values are silently discarded.

### Other pipelines

- `main_self_novelty.py` Stage 2: Correctly prints both `ip_mean` and `ip_std` — no change needed.
- `main_e2e.py`, `main_staged.py`, `main_three_staged.py`, `main_indist.py`: Check each pipeline's training loop print statements and ensure `ip_std` is printed wherever `ip_mean` or diversity loss appears.

---

## 2. Silent Failure Patterns

These patterns mask bugs by silently falling back to defaults instead of failing loudly when something is wrong.

### Unnecessary `getattr` with defaults on args that always exist

- **main_staged.py, main_three_staged.py, main_e2e.py, main_self_novelty.py, main_adversarial_distillation.py, main_indist.py:**
  `getattr(args, "use_cross_attn_routing", False)` appears across most pipelines. Since `use_cross_attn_routing` is always defined in the argparser, this should be `args.use_cross_attn_routing`.

- **main_staged.py (~line 1226):**
  `getattr(args, 'backbone', 'vanilla_gt')` — `backbone` is always defined in the argparser. Should be `args.backbone`.

### Unnecessary `hasattr` checks on model attributes

- **main_staged.py, main_three_staged.py:**
  `hasattr(model, 'multi_point_proxy')` — The model's `multi_point_proxy` attribute is always set (to either a `MultiPointProxyWrapper` instance or `None`). Use `if model.multi_point_proxy is not None:` instead.

- **main_indist.py Phase 5 (~lines 1644, 1651):**
  `hasattr(model.multi_point_proxy, '_last_proxies')` — **This is dead code.** The `MultiPointProxyWrapper` class in `generators.py` never stores a `_last_proxies` attribute. This means the novelty loss and diversity loss blocks inside the multi-point branch of Phase 5 are **never executed**. This is a bug, not just a style issue (see Section 3).

### Bare `except:` clauses in models.py

- **models.py (~lines 262–270 and ~lines 802–810):**
  Two bare `except:` clauses silently catch errors during global pooling operations. These will swallow `RuntimeError`, `TypeError`, `KeyError`, etc., making debugging extremely difficult. At minimum, catch a specific exception (e.g., `except (RuntimeError, IndexError):`) and log a warning.

---

## 3. Bugs, Logic Errors, and Implementation Issues

### main_indist.py — Missing import (runtime crash)

- **~line 521:** The function `distribution_reconstruction_loss` calls `F.normalize(...)`, but `torch.nn.functional` is never imported as `F` in this file. This will raise a `NameError` at runtime whenever this loss is used. Add `import torch.nn.functional as F` at the top of the file.

### main_indist.py — Phase 5 multi-point novelty/diversity loss is dead code

- **~lines 1640–1670:** The multi-point proxy branch computes novelty loss and diversity loss only if `hasattr(model.multi_point_proxy, '_last_proxies')` is True. Since `MultiPointProxyWrapper` in `generators.py` never stores `_last_proxies`, this entire branch is dead code. The fix requires either:
  1. Storing `_last_proxies` on the wrapper after each forward pass in `generators.py`, or
  2. Retrieving proxies through the wrapper's actual interface (e.g., from the forward return values).

### main_e2e.py — Double computation of diversity loss

- **~lines 644 and 662:** `proxy_diversity_loss(proxy_emb)` is computed twice per training iteration — once at line 644 (added to the total loss) and again at line 662 (for logging). Since this involves pairwise cosine similarity computation, it's wasteful. Store the result in a variable and reuse it.

### main_staged.py — Stage 3 aux_loss path hardcodes vanilla_gt forward interface

- **~line 1226:** When `aux_loss is not None`, the code calls:
  ```python
  model(pyg_batch, proxy_embeddings=proxy_emb, precomputed_dense=dense_batch)
  ```
  This calling convention only works for `vanilla_gt`. For `gred` and `hybrid` backbones, the forward signature is different (e.g., GRED needs distance masks and doesn't accept `precomputed_dense`). This will either crash or produce wrong results when `args.backbone != 'vanilla_gt'`.

### main_three_staged.py — Stage 3 proxy dropout doesn't disable injection

- **Stage 3 multi-point proxy dropout path:** When randomly dropping proxies during training, the code creates a subset of proxy embeddings but does **not** pass `disable_proxy_injection=True` to the model forward call. This means the model's internal proxy injection mechanism still runs, effectively injecting proxies twice (the passed subset + the internally generated ones). The dropout path should pass `disable_proxy_injection=True` so only the explicitly provided proxy subset is used.

### main_self_novelty.py — Stage 1 random proxy injection pools noise

- **Stage 1:** Random proxy embeddings (from `torch.randn`) are injected with `readout_scope="all_tokens"`. This means the model's readout pooling will include these random noise vectors alongside real node embeddings when computing graph-level representations for classification. This contaminates the classification signal during Stage 1 pretraining. Consider using `readout_scope="graph_tokens_only"` or not injecting random proxies at all during the pretraining stage.

### main_indist.py — Phase 3 novelty loss tensor shape mismatch risk

- **Phase 3 novelty loss computation:** When computing cosine similarity between proxy embeddings and node embeddings, the proxy tensor may be dense-batched (shape `[B, M, D]`) while node embeddings may be flat PyG-batched (shape `[N_total, D]`). If the reshaping/broadcasting isn't handled correctly, the cosine similarity computation will either error out or produce meaningless values. Verify that both tensors are in compatible shapes before the similarity computation.
