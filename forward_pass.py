import torch

def forward_pass(model, batch, device, config, loader, split):
    """
    Executes the forward pass for Graphormer model, handling device transfer and mask generation.
    
    Args:
        model: The Graphormer model.
        batch: The input batch dictionary.
        device: The device to move tensors to.
        config: Configuration object containing dataset_name and current_step.
        loader: The data loader (used to access dataset masks).
        split: One of "train", "val", "test".
        log_step: Optional step number for logging.
    
    Returns:
        outputs: Model outputs.
        labels: Ground truth labels.
        node_mask: The mask used for the nodes.
    """
    log_step = getattr(config, "current_step", None)
    # Move batch to device
    for k in batch:
        try:
            batch[k] = batch[k].to(device)
        except:
            batch[k] = [i.to(device) for i in batch[k]]
            
    labels = batch["labels"]
    
    # Determine node mask based on split
    mask_name = f"{split}_mask"
    node_mask = getattr(loader.dataset[0], mask_name, None)

    if split == "train" and config.dataset_name not in ["pcqm4mv2"]:
        assert node_mask is not None

    if node_mask is not None:
        node_mask = node_mask.view(-1) & ~torch.isnan(labels.view(-1))
        node_mask = node_mask.to(device)
    else:
        node_mask = torch.ones(labels.shape, dtype=torch.int32, device=device)        

    outputs = model(**batch, node_mask=node_mask, log_step=log_step, log_group=split)
    
    return outputs, labels, node_mask
