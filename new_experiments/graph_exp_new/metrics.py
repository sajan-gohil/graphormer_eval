# metrics.py
"""
Graph metrics + task abstraction.

Original: macro-AP for Peptides-func.
Extended: a small ``Task`` object (built from ``data.get_dataset_info``) that
encapsulates the per-dataset differences in loss, prediction formatting,
label numpy conversion, and metric. Training scripts get one ``Task`` from
``--dataset`` and replace every hardcoded ``BCEWithLogitsLoss`` /
``compute_macro_ap`` / ``torch.sigmoid`` site with task methods so adding
a new LRGB dataset never requires editing the training scripts again.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from data import get_dataset_info


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


# ================================================================
# DATASET-AGNOSTIC METRICS
# ================================================================

def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error — LRGB Peptides-struct standard metric."""
    if y_pred.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_pred.astype(np.float64)
                                - y_true.astype(np.float64))))


def compute_node_f1_macro(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Macro-averaged F1 over classes for per-node predictions.

    LRGB PascalVOC-SP standard metric. ``y_pred`` is expected to be argmaxed
    class indices, shape (N,). ``y_true`` is shape (N,) long. Padding
    positions should already be removed (or use ``ignore_index=-1`` upstream).
    """
    if y_pred.size == 0:
        return 0.0
    valid = y_true >= 0
    if valid.sum() == 0:
        return 0.0
    return float(f1_score(y_true[valid], y_pred[valid],
                          average="macro", zero_division=0))


def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray, ignore_index=None) -> float:
    """Accuracy for graph- or node-level multiclass tasks."""
    if y_pred.size == 0:
        return 0.0
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    if ignore_index is not None:
        mask = y_true != ignore_index
        if mask.sum() == 0:
            return 0.0
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    return float(np.mean(y_pred == y_true))


# ================================================================
# TASK ABSTRACTION
# ================================================================

class Task:
    """Per-dataset task helpers driven by the dataset registry.

    A ``Task`` packages everything that varies between LRGB datasets so the
    training scripts stay dataset-agnostic. Resolve once from ``args.dataset``:

        task = Task(args.dataset)

    Then replace hardcoded sites:
        nn.BCEWithLogitsLoss()                      -> task.loss_fn
        loss_fn(logits, batch.y)                    -> task.loss(logits, batch.y)
        torch.sigmoid(logits).detach().cpu().numpy() -> task.predict(logits)
        batch.y.cpu().numpy()                        -> task.labels_to_numpy(batch.y)
        compute_macro_ap(preds, labels)             -> task.compute_metric(preds, labels)
    """

    def __init__(self, dataset_name, dataset_info=None, pos_weight=None):
        info = dataset_info or get_dataset_info(dataset_name)
        self.dataset_name = info.get("name", dataset_name)
        self.output_dim = info["output_dim"]
        self.task_type = info["task_type"]   # multi_label | regression | multiclass
        self.level = info["level"]           # graph | node
        self.node_encoder = info["node_encoder"]
        self.node_feat_dim = info["node_feat_dim"]
        self.metric_name = info["metric_name"]

        if self.task_type == "multi_label":
            # pos_weight: optional (C,) tensor of per-class positive weights,
            # registered as a buffer inside BCEWithLogitsLoss so .to(device)
            # moves it with the module.
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.task_type == "regression":
            # LRGB Peptides-struct uses L1 / MAE.
            self.loss_fn = nn.L1Loss()
        elif self.task_type == "multiclass":
            # ignore_index=-1 leaves room for masked padding in node targets.
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _flatten_labels(self, y):
        """Flatten labels for multiclass tasks to match (N,) expectations.

        CrossEntropyLoss expects 1D class indices when logits are 2D
        (batch_or_nodes, num_classes). Some datasets store labels with an
        extra singleton dimension, so flattening keeps loss/metrics aligned.
        """
        print("METRICS _flatten_label ===================== ", y.dim())
        if y.dim() > 1:
            return y.view(-1)
        return y

    # --- Loss --------------------------------------------------------
    def loss(self, logits, y):
        """Compute the task loss with the right dtype/shape coercion.

        Expected shapes:
          - multi_label:  logits (B, K) float;   y (B, K) float-or-bool
          - regression:   logits (B, K) float;   y (B, K) float
          - multiclass:   logits (N, K) float;   y (N,) long (per-node tasks)
        """
        if self.task_type == "multi_label":
            return self.loss_fn(logits, y.float())
        if self.task_type == "regression":
            return self.loss_fn(logits, y.float())
        if self.task_type == "multiclass":
            y = y.long()
            y = self._flatten_labels(y)
            return self.loss_fn(logits, y)
        raise ValueError(f"Unknown task_type: {self.task_type}")

    # --- Numpy adapters for metric accumulation ---------------------
    def predict(self, logits) -> np.ndarray:
        """Convert raw logits into the numpy format ``compute_metric`` expects."""
        if self.task_type == "multi_label":
            return torch.sigmoid(logits).detach().cpu().numpy()
        if self.task_type == "regression":
            return logits.detach().cpu().numpy()
        if self.task_type == "multiclass":
            return logits.argmax(dim=-1).detach().cpu().numpy()
        raise ValueError(f"Unknown task_type: {self.task_type}")

    def labels_to_numpy(self, y) -> np.ndarray:
        """Convert ground-truth tensors to numpy aligned with ``predict``."""
        #if self.task_type == "multiclass":
        #    return self._flatten_labels(y).detach().cpu().numpy()
        return y.detach().cpu().numpy()

    # --- Final metric -----------------------------------------------
    def compute_metric(self, preds: np.ndarray, labels: np.ndarray) -> float:
        if self.metric_name == "macro_ap":
            return compute_macro_ap(preds, labels)
        if self.metric_name == "mae":
            return compute_mae(preds, labels)
        if self.metric_name == "node_f1_macro":
            return compute_node_f1_macro(preds, labels)
        if self.metric_name == "accuracy":
            ignore_index = -1 if self.level == "node" else None
            return compute_accuracy(preds, labels, ignore_index=ignore_index)
        raise ValueError(f"Unknown metric_name: {self.metric_name}")

    # --- Friendlier metric direction (higher_is_better) -------------
    @property
    def higher_is_better(self) -> bool:
        # AP and F1 are higher-is-better; MAE is lower-is-better.
        return self.metric_name in ("macro_ap", "node_f1_macro", "accuracy")

    @property
    def metric_label(self) -> str:
        return {
            "macro_ap": "AP",
            "mae": "MAE",
            "node_f1_macro": "F1",
            "accuracy": "Acc",
        }[self.metric_name]


def build_task(dataset_name, dataset_info=None, pos_weight=None) -> Task:
    """Convenience constructor — used at the top of every training script."""
    return Task(dataset_name, dataset_info=dataset_info, pos_weight=pos_weight)


def compute_pos_weight(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class positive weight = sqrt(N / (C * n_k)).

    N = number of samples, C = number of classes, n_k = number of positive
    samples for class k. Classes with no positives fall back to n_k = 1 to
    avoid division by zero.

    labels: (N, C) binary array of multi-label targets.
    Returns: (C,) float array of pos_weight values.
    """
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, num_classes)
    N = labels.shape[0]
    n_k = labels.sum(axis=0).astype(np.float64)         # positives per class
    n_k = np.clip(n_k, 1.0, None)
    return np.sqrt(N / (num_classes * n_k))

