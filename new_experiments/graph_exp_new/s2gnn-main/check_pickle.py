import pickle
import numpy as np

path = 'datasets/peptides_functional/dist_masks/all_max_hops_30.pkl'
with open(path, 'rb') as f:
    masks = pickle.load(f)

print(f"Loaded {len(masks)} masks.")
for i in range(10):
    print(f"Graph {i} mask shape: {masks[i].shape}")

# Also check max K across all masks
max_k = max(m.shape[0] for m in masks)
print(f"Max K across all graphs: {max_k}")
