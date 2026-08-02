from data import get_dataset_info, get_loaders

loaders = get_loaders(dataset_name="Peptides-func", use_dist_masks=True, max_hops=30)
train_loader = loaders[0]
batch, dist_masks, node_masks = next(iter(train_loader))
print(f"Dist masks shape: {dist_masks.shape}")

# Also let's check the actual distances
for i in range(min(5, dist_masks.shape[0])):
    dm = dist_masks[i] # (max_hops, max_N, max_N)
    valid_k = []
    for k in range(dm.shape[0]):
        if dm[k].sum() > 0:
            valid_k.append(k)
    print(f"Graph {i} has valid hops: {valid_k}")
