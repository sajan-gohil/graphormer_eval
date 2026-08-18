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
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from data import get_dataset_info


# ================================================================
# FOCAL LOSS
# ================================================================

class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for multi-label binary classification.

    Implements FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) in a
    numerically stable way by computing standard BCE first, then
    applying the down-weighting factor for easy examples.

    Args:
        gamma:      focusing parameter (≥ 0).  gamma=0 recovers standard BCE.
                    Typical values: 0.5, 1, 2.  Higher gamma down-weights
                    easy (high-confidence) examples more aggressively.
        pos_weight: optional (C,) tensor — per-class multiplicative weight
                    on positive examples.  Applied on top of the focal weight
                    (orthogonal to gamma; can combine with --use_pos_weight).
        reduction:  'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        # ── Numerically stable BCE per element ───────────────────────
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # ── p_t: probability of the "correct" outcome ─────────────────
        # p_t = sigmoid(logit) for positives, 1-sigmoid for negatives
        with torch.no_grad():
            p = torch.sigmoid(logits)
            p_t = p * targets + (1.0 - p) * (1.0 - targets)

        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * bce

        # ── Optional pos_weight (alpha) ───────────────────────────────
        # Scale positive-example loss by pos_weight[c]; negatives unscaled.
        if self.pos_weight is not None:
            alpha_t = self.pos_weight.unsqueeze(0) * targets + (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalCrossEntropyLoss(nn.Module):
    """Focal loss for multiclass node/graph classification.

    Wraps standard cross-entropy and applies the (1 - p_t)^gamma factor so
    the gradient on confidently-correct predictions is suppressed.

    Args:
        gamma:        focusing parameter (≥ 0).  0 = standard CE.
        ignore_index: target value to ignore (default -1 for padded nodes).
        reduction:    'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-element CE, ignore_index handled by masking below
        ce = F.cross_entropy(
            logits, targets, ignore_index=self.ignore_index, reduction="none"
        )

        valid = targets != self.ignore_index   # (N,) or (B,) mask

        with torch.no_grad():
            # p_t = probability the model assigned to the correct class
            p_t = torch.exp(-ce)

        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * ce

        if self.reduction == "mean":
            denom = valid.sum()
            return loss[valid].sum() / denom.clamp(min=1) if denom > 0 else loss.new_tensor(0.0)
        if self.reduction == "sum":
            return loss[valid].sum()
        return loss


def compute_macro_ap(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute macro-averaged AP over all classes (LRGB / OGB protocol).

    NaN entries in ``y_true`` are treated as "not measured" and skipped, which
    is what ogbg-molpcba requires: most (molecule, task) pairs are unlabelled,
    and ``average_precision_score`` raises on NaN. Columns with no positive or
    no negative example are skipped, matching OGB's evaluator.

    Args:
        y_pred: (N, C) predicted probabilities (after sigmoid).
        y_true: (N, C) binary ground truth labels, possibly containing NaN.

    Returns:
        Macro-averaged AP over the columns that could be scored.
    """
    if len(y_pred) == 0:
        return 0.0

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    if not np.isnan(y_true).any():
        return float(average_precision_score(y_true, y_pred, average="macro"))

    scores = []
    for c in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, c])
        col = y_true[valid, c]
        # Need both classes present for AP to be defined.
        if col.size == 0 or col.sum() == 0 or col.sum() == col.size:
            continue
        scores.append(average_precision_score(col, y_pred[valid, c]))
    return float(np.mean(scores)) if scores else 0.0


def compute_rocauc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Macro-averaged ROC-AUC — ogbg-molhiv standard metric.

    Same NaN handling as ``compute_macro_ap`` so it also works for multi-task
    OGB molecule datasets.
    """
    if len(y_pred) == 0:
        return 0.0

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    scores = []
    for c in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, c])
        col = y_true[valid, c]
        if col.size == 0 or col.sum() == 0 or col.sum() == col.size:
            continue
        scores.append(roc_auc_score(col, y_pred[valid, c]))
    return float(np.mean(scores)) if scores else 0.0


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

    def __init__(self, dataset_name, dataset_info=None, pos_weight=None,
                 focal_gamma: float = 0.0, label_smoothing: float = 0.0):
        info = dataset_info or get_dataset_info(dataset_name)
        self.dataset_name = info.get("name", dataset_name)
        self.output_dim = info["output_dim"]
        self.task_type = info["task_type"]   # multi_label | regression | multiclass
        self.level = info["level"]           # graph | node
        self.node_encoder = info["node_encoder"]
        self.node_feat_dim = info["node_feat_dim"]
        self.metric_name = info["metric_name"]
        # ogbg-molpcba leaves most (molecule, task) pairs unmeasured as NaN.
        # Those positions must be excluded from the loss, not coerced to 0.
        self.nan_labels = bool(info.get("nan_labels", False))
        self.focal_gamma = focal_gamma

        if self.level == "link":
            raise NotImplementedError(
                f"{self.dataset_name} is a link-level task; Task supports "
                f"'graph' and 'node' only."
            )
        # Label smoothing replaces hard {0,1} targets with {ε, 1−ε} for
        # multi_label tasks and passes label_smoothing to CrossEntropyLoss for
        # multiclass tasks.  Has no effect on regression.
        self.label_smoothing = max(0.0, label_smoothing)

        _use_focal = focal_gamma > 0.0

        if self.task_type == "multi_label":
            # pos_weight: optional (C,) tensor of per-class positive weights,
            # registered as a buffer inside BCEWithLogitsLoss so .to(device)
            # moves it with the module.
            # Label smoothing is applied at the target level in Task.loss(),
            # so the loss_fn itself is unchanged.
            if _use_focal:
                self.loss_fn = FocalBCEWithLogitsLoss(
                    gamma=focal_gamma, pos_weight=pos_weight
                )
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.task_type == "regression":
            # LRGB Peptides-struct uses L1 / MAE.  Neither focal loss nor label
            # smoothing is meaningful for regression; both are silently ignored.
            self.loss_fn = nn.L1Loss()
        elif self.task_type == "multiclass":
            # ignore_index=-1 leaves room for masked padding in node targets.
            # PyTorch's CrossEntropyLoss has native label_smoothing support;
            # FocalCrossEntropyLoss uses hard targets so smoothing is skipped
            # (focal + smoothing interact non-trivially — handle separately).
            if _use_focal:
                if label_smoothing > 0.0:
                    print(f"[Task] label_smoothing={label_smoothing} ignored for "
                          f"multiclass + focal_gamma={focal_gamma}: focal and label "
                          f"smoothing interact non-trivially; use one at a time.",
                          flush=True)
                self.loss_fn = FocalCrossEntropyLoss(
                    gamma=focal_gamma, ignore_index=-1
                )
            else:
                self.loss_fn = nn.CrossEntropyLoss(
                    ignore_index=-1, label_smoothing=self.label_smoothing
                )
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

        Label smoothing (multi_label only):
          Hard binary targets {0, 1} are replaced with soft targets {ε, 1−ε}:
              y_smooth = y * (1 − 2ε) + ε
          This prevents the model from driving BCE to zero on training positives,
          compressing the train-val confidence gap without changing the metric.
          Works identically with and without focal loss since it operates on the
          target tensor before any loss computation.
        """
        if self.task_type == "multi_label":
            y_f = y.float()
            if logits.shape != y_f.shape:
                # OGB stores graph targets as (1, T) per graph, so a batch
                # arrives as (B, T) already; guard the (B*T,) edge case.
                y_f = y_f.view(logits.shape)
            if self.nan_labels:
                # Reduce over measured entries only. Substituting a finite value
                # at NaN positions keeps the graph clean; the mask then zeroes
                # their contribution before the mean.
                valid = ~torch.isnan(y_f)
                y_safe = torch.where(valid, y_f, torch.zeros_like(y_f))
                if self.label_smoothing > 0.0:
                    eps = self.label_smoothing
                    y_safe = y_safe * (1.0 - 2.0 * eps) + eps
                per_el = F.binary_cross_entropy_with_logits(
                    logits, y_safe, reduction="none")
                if getattr(self.loss_fn, "pos_weight", None) is not None:
                    pw = self.loss_fn.pos_weight.unsqueeze(0)
                    per_el = per_el * (pw * y_safe + (1.0 - y_safe))
                denom = valid.sum()
                if denom == 0:
                    return logits.sum() * 0.0
                return (per_el * valid).sum() / denom
            if self.label_smoothing > 0.0:
                eps = self.label_smoothing
                y_f = y_f * (1.0 - 2.0 * eps) + eps
            return self.loss_fn(logits, y_f)
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
        if self.metric_name == "rocauc":
            return compute_rocauc(preds, labels)
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
        # AP, F1, accuracy and ROC-AUC are higher-is-better; MAE is lower.
        return self.metric_name in ("macro_ap", "node_f1_macro", "accuracy",
                                    "rocauc")

    @property
    def metric_label(self) -> str:
        return {
            "macro_ap": "AP",
            "mae": "MAE",
            "node_f1_macro": "F1",
            "accuracy": "Acc",
            "rocauc": "ROC-AUC",
        }[self.metric_name]


def build_task(dataset_name, dataset_info=None, pos_weight=None,
               focal_gamma: float = 0.0, label_smoothing: float = 0.0) -> Task:
    """Convenience constructor — used at the top of every training script."""
    return Task(dataset_name, dataset_info=dataset_info, pos_weight=pos_weight,
                focal_gamma=focal_gamma, label_smoothing=label_smoothing)


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
