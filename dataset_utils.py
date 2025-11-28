import os
from collections import namedtuple
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List
from torch.utils.data import Subset
from graphormer_hf.collating_graphormer import GraphormerDataCollator
# from cobformer_data.get_data import get_data
# from cobformer_data.data_utils import load_fixed_splits

cobformer_datasets_n_patches = {
    "cora": 112,
    "citeseer": 144,
    "pubmed": 224,
    "film": 112,
    "deezer": 224,
    "ogbn-arxiv": 2048,
    "ogbn-products": 8192,
}

COBFORMER_DATASETS = {"cora", "citeseer", "pubmed", "film", "deezer", "ogbn-arxiv", "ogbn-products"}


def load_data(dataset_name, num_workers=0, batch_size=512, config=None):
    if dataset_name in COBFORMER_DATASETS:
        # Use get_data from cobformer_data for consistent processing
        path = f"datasets/{dataset_name}"
        data = torch.load(f"{path}/{dataset_name}.pt", weights_only=False)
        # Splitting/partitioning logic
        # n_patch = cobformer_datasets_n_patches[dataset_name] 
        # patch = data.partition_patch(n_patch)
        # print(data, split_dict, patch)
        # dataloader
        train_collate_fn = GraphormerDataCollator(on_the_fly_processing=True, config=config, split="train")
        train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=train_collate_fn)
        if not config.augment_edges and not config.create_subgraph:
            val_collate_fn = train_collate_fn
            val_loader = test_loader = train_loader
        else:
            val_collate_fn = GraphormerDataCollator(on_the_fly_processing=True, config=config, split="val")
            val_loader = test_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=val_collate_fn)
        return train_loader, val_loader, test_loader

    else:
        data_path = f"datasets/{dataset_name}/{dataset_name}.pt"
        split_path = f"datasets/{dataset_name}/split_dict.pt"
        data = torch.load(data_path, weights_only=False)
        if os.path.exists(split_path):
            split_dict = torch.load(split_path, weights_only=False)
            train_idx = split_dict['train']
            valid_idx = split_dict['valid']
            test_idx = split_dict.get('test')
        else:  # split 50:25:25 if no split file exists
            num_nodes = len(data)
            indices = torch.randperm(num_nodes)
            train_idx = indices[:int(0.5 * num_nodes)]
            valid_idx = indices[int(0.5 * num_nodes):int(0.75 * num_nodes)]
            test_idx = indices[int(0.75 * num_nodes):]

        train_dataset = Subset(data, train_idx[:len(train_idx) // 10])  # Use a smaller subset for faster training
        val_dataset = Subset(data, valid_idx)
        test_dataset = Subset(data, test_idx)

        collate_fn = GraphormerDataCollator(on_the_fly_processing=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    dataset_name = "cora"  # Change to your dataset name
    train_loader, val_loader, test_loader = load_data(dataset_name)
    print(f"Loaded {dataset_name} with {len(train_loader.dataset)} training samples.")
