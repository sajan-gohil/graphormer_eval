"""
Phase 1 Metrics: AP computation compatible with Peptides-func / LRGB evaluation protocol.
"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score


def compute_macro_ap(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute macro-averaged AP over all classes (LRGB protocol).

    Args:
        y_pred: (N, C) predicted probabilities (after sigmoid).
        y_true: (N, C) binary ground truth labels.

    Returns:
        Macro-averaged AP score.
    """
    # Handle edge cases
    if len(y_pred) == 0:
        return 0.0
    return average_precision_score(y_true, y_pred, average='macro')


def compute_per_class_ap(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute per-class AP scores.

    Args:
        y_pred: (N, C) predicted probabilities.
        y_true: (N, C) binary ground truth labels.

    Returns:
        Array of per-class AP scores, shape (C,).
    """
    num_classes = y_true.shape[1]
    per_class = np.zeros(num_classes)
    for c in range(num_classes):
        if y_true[:, c].sum() > 0:
            per_class[c] = average_precision_score(y_true[:, c], y_pred[:, c])
        else:
            per_class[c] = float('nan')
    return per_class


def compute_correct_classes(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    For each sample, return the set of classes correctly predicted (threshold 0.5).
    Used later for the "at least one more class" filtering criterion.

    Args:
        y_pred: (N, C) predicted probabilities.
        y_true: (N, C) binary ground truth labels.
        threshold: Classification threshold.

    Returns:
        List of sets, where each set contains the indices of correctly predicted classes.
    """
    pred_binary = (y_pred >= threshold).astype(int)
    correct = (pred_binary == y_true).astype(int)
    result = []
    for i in range(len(y_true)):
        correct_classes = set()
        for c in range(y_true.shape[1]):
            if correct[i, c] == 1:
                correct_classes.add(c)
        result.append(correct_classes)
    return result


@torch.no_grad()
def collect_predictions(model, loader, device, proxy_fn=None):
    """
    Run model on a dataloader and collect all predictions and labels.

    Handles models that return (logits, node_embeddings) tuples.

    Args:
        model: GraphTransformer or similar.
        loader: DataLoader.
        device: torch device.
        proxy_fn: optional callable(batch) -> (B, M, d) proxy embeddings.

    Returns:
        y_pred: np.ndarray of shape (N, C) — sigmoid probabilities
        y_true: np.ndarray of shape (N, C) — binary labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        proxies = proxy_fn(batch) if proxy_fn else None
        out = model(batch, proxy_embeddings=proxies)
        # Handle (logits, node_emb) tuple or plain logits
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_pred, y_true


def evaluate(model, loader, device, proxy_fn=None):
    """
    Evaluate model on a loader and return (macro_AP, mean_loss).

    Args:
        model: GraphTransformer or similar.
        loader: DataLoader.
        device: torch device.
        proxy_fn: optional callable(batch) -> (B, M, d) proxy embeddings.

    Returns:
        (ap, loss): macro-averaged AP and mean BCE loss.
    """
    import torch.nn as nn
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    all_preds = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            proxies = proxy_fn(batch) if proxy_fn else None
            out = model(batch, proxy_embeddings=proxies)
            logits = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logits, batch.y)
            losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    ap = compute_macro_ap(y_pred, y_true)
    return ap, float(np.mean(losses))
