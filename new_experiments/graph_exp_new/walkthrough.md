# Walkthrough: S2GNN Spectral Filter Alignment

## What Changed

Both model files were updated so their spectral computation matches S2GNN's pipeline from [s2_filter_encoder.py](file:///wsl$/Ubuntu/home/srg/graphormer_eval/new_experiments/graph_exp_new/s2gnn-main/graphgps/layer/s2_filter_encoder.py) and [s2_spectral.py](file:///wsl$/Ubuntu/home/srg/graphormer_eval/new_experiments/graph_exp_new/s2gnn-main/graphgps/layer/s2_spectral.py).

---

### [model_hop_masked_transformer_spectral.py](file:///wsl$/Ubuntu/home/srg/graphormer_eval/new_experiments/graph_exp_new/model_hop_masked_transformer_spectral.py)

#### New: `GaussianSmearing` class (line ~50)
Inline implementation matching PyG's `torch_geometric.nn.models.schnet.GaussianSmearing`. Maps scalar eigenvalues to a vector of `num_gaussians` evenly-spaced Gaussian basis values.

#### Modified: `SpectralBandFilter` (line ~353)

| Component | Before | After (S2GNN-style) |
|---|---|---|
| **Filter parameterization** | Hand-rolled RBF `exp(-0.5·((λ-c)/w)²)` with learnable 3D coefficient tensor `[S, R, Dh]` combined via `einsum("bkr,srd->bksd")` | `GaussianSmearing(0, 0.7, 60)` → bottleneck MLP `Linear(60→1, no bias) → Linear(1→S)` producing per-band filter `[B, K, S]`, broadcast over `Dh` |
| **Windowing** | None | Tukey (tapered-cosine) window with `α=0.5`, `cutoff=0.7`. Full pass for λ ≤ 0.35, cosine taper to zero for λ ∈ (0.35, 0.7], hard zero beyond 0.7 |
| **Feature transform** | None — raw features fed to `in_proj` | Simplified GLU gate: `F.silu(Linear(d→0.05d→d)) * x` before `in_proj` (matching S2GNN's `SpecFeatureTransformLayer` with `glu_0.05`) |
| **Default K** | 16 eigenpairs | 150 eigenpairs (matching S2GNN's `max_freqs: 150`) |

New configurable parameters threaded through `HopMaskedTransformerLayer` → `HopMaskedTransformerModel`:

| Parameter | Default | S2GNN config equivalent |
|---|---|---|
| `num_gaussians` | 60 | `basis_num_gaussians: 60` |
| `basis_bottleneck` | 0.2 | `basis_bottleneck: 0.2` |
| `frequency_cutoff` | 0.7 | `frequency_cutoff: 0.7` |
| `tukey_alpha` | 0.5 | `window: tukey` (default α=0.5) |
| `glu_bottleneck` | 0.05 | `feature_transform: glu_0.05` |
| `spectral_k` | 150 | `max_freqs: 150` |

---

### [model_hop_masked_transformer_spectral_hop.py](file:///wsl$/Ubuntu/home/srg/graphormer_eval/new_experiments/graph_exp_new/model_hop_masked_transformer_spectral_hop.py)

#### New: `GaussianSmearing` class (line ~50)
Same inline implementation as above.

#### Modified: `SpectralCrossHopMixer` (line ~354)

| Component | Before | After (S2GNN-style) |
|---|---|---|
| **Filter parameterization** | Simple learnable parameter `spectral_response = zeros(H, Dh)` — one weight per mode per channel | `GaussianSmearing(0, 2, 60)` → bottleneck MLP `Linear(60→bn, no bias) → Linear(bn→Dh)` applied to the fixed hop eigenvalues |
| **Windowing** | None | Tukey window with `α=0.5` and 2.5% wiggle beyond `λ_max` (matching S2GNN's default `wiggle=0.025`) |
| **GLU** | Not applicable (hop-axis filter, not feature-level) | Not added (intentional — the input is already processed node features) |

> [!NOTE]
> The `SpectralCrossHopMixer` operates on the *hop/head axis* (a fixed path-graph eigenbasis), not the graph Laplacian. The S2GNN-style filter parameterization (GaussianSmearing + MLP) replaces the old simple learnable `spectral_response` parameter, giving a richer but structured filter. The Tukey window ensures smooth spectral truncation.

## Verification
- ✅ Both files pass Python syntax validation (`ast.parse`)
