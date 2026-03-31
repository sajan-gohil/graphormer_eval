import pickle
import torch
import torch_geometric
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader


def get_loaders(batch_size=256, num_workers=4):
    """Load Peptides-func train/val/test splits and return loaders + datasets."""
    train_ds = LRGBDataset(root="./data", name="Peptides-func", split="train")
    val_ds = LRGBDataset(root="./data", name="Peptides-func", split="val")
    test_ds = LRGBDataset(root="./data", name="Peptides-func", split="test")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        train_ds, val_ds, test_ds,
    )


class ProxyTargetDataset(torch.utils.data.Dataset):
    """
    Pairs each graph with its best optimized proxy embedding set.

    Loads proxy pairs from a pickle file produced by Stage 2. For each graph
    (identified by sample_idx), keeps the entry with the lowest opt_loss
    across all optimization restarts.

    Graphs without any valid proxy pair are excluded.
    """
    def __init__(self, pyg_dataset, proxy_pairs_path):
        with open(proxy_pairs_path, "rb") as f:
            proxy_pairs = pickle.load(f)

        # Find best proxy per sample_idx (lowest opt_loss across restarts)
        best_by_idx = {}
        for p in proxy_pairs:
            sid = p["sample_idx"]
            if sid not in best_by_idx or p["opt_loss"] < best_by_idx[sid]["opt_loss"]:
                best_by_idx[sid] = p

        # Build aligned lists — only include graphs that have a valid proxy
        self.samples = []
        for sid in sorted(best_by_idx.keys()):
            pair = best_by_idx[sid]
            self.samples.append({
                "graph": pyg_dataset[sid],
                "encoder_emb": pair["encoder_emb"],      # (max_N, d)
                "mask": pair["mask"],                      # (max_N,)
                "proxy_emb": pair["proxy_emb"],            # (M, d)
                "sample_idx": sid,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["graph"], s["encoder_emb"], s["mask"], s["proxy_emb"]


def collate_with_proxies(batch):
    """
    Collate function for ProxyTargetDataset.

    Returns:
        pyg_batch: batched PyG Data object
        encoder_embs: (B, max_N, d) padded encoder embeddings
        emb_masks: (B, max_N) boolean masks
        proxy_embs: (B, M, d) target proxy embeddings
    """
    graphs, encoder_embs, masks, proxy_embs = zip(*batch)

    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))
    proxy_batch = torch.stack(proxy_embs, dim=0)

    # Pad encoder embeddings and masks to same max_N within this batch
    max_n = max(e.shape[0] for e in encoder_embs)
    d = encoder_embs[0].shape[1]
    B = len(encoder_embs)

    padded_embs = torch.zeros(B, max_n, d)
    padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (emb, mask) in enumerate(zip(encoder_embs, masks)):
        n = emb.shape[0]
        padded_embs[i, :n] = emb
        padded_masks[i, :n] = mask

    return pyg_batch, padded_embs, padded_masks, proxy_batch
