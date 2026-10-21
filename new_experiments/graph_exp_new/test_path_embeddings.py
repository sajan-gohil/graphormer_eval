# test_path_embeddings.py
"""Checks PathEmbedder against an explicit per-path loop, and smoke-tests both
attention modes (hop-masked / multihop) with path embeddings + cross-hop tags.
Run: python test_path_embeddings.py
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

from model_hop_masked_light import HopMaskedTransformerModel

torch.manual_seed(0)
K, D, H = 6, 32, 4


def make_graph(n):
    ei = torch.stack([torch.arange(n - 1), torch.arange(1, n)])          # path
    ei = torch.cat([ei, torch.tensor([[0], [n // 2]])], 1)                # + chord
    ei = to_undirected(ei)
    return Data(x=torch.randint(0, 2, (n, 9)), edge_index=ei,
                edge_attr=torch.randint(0, 2, (ei.shape[1], 3)), num_nodes=n)


def dist_pred(g):
    n = g.num_nodes
    a = csr_matrix((np.ones(g.edge_index.shape[1]), g.edge_index.numpy()), (n, n))
    d, p = floyd_warshall(a, directed=False, unweighted=True, return_predecessors=True)
    d[np.isinf(d)] = 10_000
    return d.astype(int), np.where(p < 0, -1, p)


graphs = [make_graph(7), make_graph(10)]
batch = Batch.from_data_list(graphs)
B, N = len(graphs), max(g.num_nodes for g in graphs)
dm = torch.zeros(B, K, N, N)
pred = torch.full((B, N, N), -1, dtype=torch.long)
nm = torch.zeros(B, N, dtype=torch.bool)
dists = []
for i, g in enumerate(graphs):
    n = g.num_nodes
    d, p = dist_pred(g)
    dists.append(d)
    for k in range(K):
        dm[i, k, :n, :n] = torch.from_numpy(d == k).float()
    pred[i, :n, :n] = torch.from_numpy(p)
    nm[i, :n] = True
batch.path_data = {"pred": pred}

for multihop in (False, True):
    model = HopMaskedTransformerModel(
        hidden_dim=D, num_heads=H, num_layers=2, max_hops=K, output_dim=3,
        dataset_name="Peptides-func", use_path_embeddings=True,
        dynamic_cross_hop=True, multihop_attn=multihop, dropout=0.0)
    model.eval()

    # --- explicit reference for P ---
    pe = model.path_embedder
    dense_x, _ = model.encode_dense(batch)
    em = model._build_edge_map(batch, B, N)
    P = pe(dense_x, batch.edge_attr, em, dm, pred, nm)
    Eemb = pe.W_e(pe.edge_act(pe.edge_enc(batch.edge_attr)))
    max_err = 0.0
    for b in range(B):
        n = graphs[b].num_nodes
        for s in range(n):
            for t in range(n):
                dst = dists[b][s, t]
                ref = pe.hop_emb.weight[dst if dst < K else K].clone()
                if 1 <= dst < K:
                    path, v = [t], t
                    while v != s:
                        v = int(pred[b, s, v]); path.append(v)
                    path = path[::-1]
                    acc = torch.zeros(D)
                    for r, (u, w) in enumerate(zip(path[:-1], path[1:])):
                        acc += (pe.W_s(dense_x[b, u]) + pe.W_t(dense_x[b, w])
                                + Eemb[em[b, u, w]] + pe.pos_emb.weight[r])
                    ref = ref + acc / dst ** 0.5
                ref = pe.norm(ref)
                max_err = max(max_err, (P[b, s, t] - ref).abs().max().item())
    assert max_err < 1e-5, max_err
    assert P[~(nm[:, :, None] & nm[:, None, :])].abs().max() == 0

    # --- forward/backward smoke test ---
    model.train()
    # Gate starts at 0: its own grad must be nonzero; P params need a nonzero gate.
    logits, *_ = model(batch, dm, nm)
    logits.sum().backward()
    assert model.layers[0].attn.path_gate_scale.grad.abs().sum() > 0, "gate grad"
    model.zero_grad()
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.path_gate_scale.fill_(1.0)
    logits, *_ = model(batch, dm, nm)
    logits.sum().backward()
    for name in ("path_embedder.pos_emb.weight", "path_embedder.hop_emb.weight",
                 "path_embedder.W_e.weight", "path_embedder.W_s.weight"):
        gr = dict(model.named_parameters())[name].grad
        assert gr is not None and gr.abs().sum() > 0, name
    print(f"multihop={multihop}: P max err {max_err:.2e}, logits {tuple(logits.shape)}, grads ok")

# --- grouped multihop softmax == per-hop loop (eval, shortest_path masks) ---
from model_hop_masked_light import MultiHopMaskedMHA

x = torch.randn(B, N, D)
P_test = torch.randn(B, N, N, D)
for readout in ("sum", "mean"):
    loop = MultiHopMaskedMHA(D, H, dropout=0.1, readout=readout, use_path_gate=True)
    grp = MultiHopMaskedMHA(D, H, dropout=0.1, readout=readout, use_path_gate=True,
                            grouped_softmax=True)
    grp.load_state_dict(loop.state_dict())
    with torch.no_grad():
        loop.path_gate_scale.fill_(0.7)
        grp.path_gate_scale.fill_(0.7)
    loop.eval(); grp.eval()
    err = (loop(x, dm, nm, P_test) - grp(x, dm, nm, P_test)).abs().max().item()
    assert err < 1e-5, (readout, err)
    print(f"grouped softmax == loop (readout={readout}): max err {err:.2e}")

# --- GPU-built hop masks == collate-built masks ---
from train_hop_masked_light import _move_batch_to_device
dist_idx = torch.full((B, N, N), -1, dtype=torch.int16)
for i, d_ in enumerate(dists):
    n = graphs[i].num_nodes
    dist_idx[i, :n, :n] = torch.from_numpy(np.where(d_ < K, d_, -1).astype(np.int16))
_, dm_built, _, _ = _move_batch_to_device((batch, dist_idx, nm), "cpu", K)
assert torch.equal(dm_built, dm), "hop masks differ"
print("device-built hop masks == collate masks")

# --- precomputed levels (no nonzero) == nonzero fallback ---
from data import _path_levels
model = HopMaskedTransformerModel(
    hidden_dim=D, num_heads=H, num_layers=1, max_hops=K, output_dim=3,
    dataset_name="Peptides-func", use_path_embeddings=True, dropout=0.0).eval()
pe = model.path_embedder
dense_x, _ = model.encode_dense(batch)
em = model._build_edge_map(batch, B, N)
levels = _path_levels(dist_idx.numpy(), pred.numpy().astype(np.int16), K - 1)
with torch.no_grad():
    P_lv = pe(dense_x, batch.edge_attr, em, dm, pred, nm, levels=levels)
    P_nz = pe(dense_x, batch.edge_attr, em, dm, pred, nm)
assert (P_lv - P_nz).abs().max().item() < 1e-6
print("levels path == nonzero path")

# --- vectorized per-head mask / blend bias == reference loops ---
def ref_mask(dmask, hop_sets, H_):
    B_, K_, N_, _ = dmask.shape
    out = torch.zeros(B_, H_, N_, N_, dtype=torch.bool)
    for h, hs in enumerate(hop_sets):
        idx = [] if hs is None else [k for k in hs if k < K_]
        out[:, h] = True if not idx else dmask[:, idx].bool().any(1)
    eye = torch.eye(N_, dtype=torch.bool)
    has = (out & ~eye).any(-1).any(-1)
    out[~has] = True
    return out

def ref_blend(adj, hop_sets, H_):
    B_, K_, N_, _ = adj.shape
    out = adj.new_zeros(B_, H_, N_, N_)
    for h, hs in enumerate(hop_sets):
        idx = [] if hs is None else [k for k in hs if 0 < k < K_]
        if idx:
            out[:, h] = adj[:, idx].sum(1)
    return out

for mask_type in ("shortest_path", "adj_power"):
    m = HopMaskedTransformerModel(
        hidden_dim=D, num_heads=H, num_layers=1, max_hops=K, output_dim=3,
        dataset_name="Peptides-func", mask_type=mask_type, num_global_heads=1,
        blend_adj_power=True, dropout=0.0)
    src = m._build_adj_power_masks(dm) if mask_type == "adj_power" else dm
    hs = m.head_hop_sets
    assert torch.equal(m._build_per_head_mask(src, hs), ref_mask(src, hs, H)), mask_type
    adj = m._build_adj_power_masks(src)
    assert (m._build_adj_blend_bias_from_power(adj, hs) - ref_blend(adj, hs, H)).abs().max() < 1e-6
    print(f"vectorized per-head mask / blend bias == loop ({mask_type})")

