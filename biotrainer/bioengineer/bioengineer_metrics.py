import numpy as np
from typing import Dict, Optional


def compute_contact_precision(
    predictions: np.ndarray,
    targets: np.ndarray,
    minsep: int = 6,
    maxsep: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute precision scores for the predicted contact map at a given range.
    Reference: https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb (reference implementation in torch; converted to numpy below)

    Args:
        predictions: Predicted contact map. [L, L]
        targets: Target contact map. [L, L]
        minsep: Minimum separation for given range.
        maxsep: Maximum separation for given range.

    Returns:
        Dict[str, float]: Dictionary containing the precision scores.
            "AUC": AUC
            "P@L": P@L
            "P@L2": P@L2
            "P@L5": P@L5
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.shape}, "
            f"targets of size {targets.shape}"
        )
    seqlen = predictions.shape[0]
    seqlen_range = np.arange(seqlen)

    sep = seqlen_range[np.newaxis, :] - seqlen_range[:, np.newaxis]  # torch equivalent: seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)
    if maxsep is not None:
        valid_mask &= sep < maxsep
    predictions = np.where(valid_mask, predictions, -np.inf)  # torch equivalent: predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[x_ind, y_ind]
    targets_upper = targets[x_ind, y_ind]

    indices = np.argsort(-predictions_upper)[:seqlen]  # torch equivalent: predictions_upper.argsort(dim=-1, descending=True)[:seqlen]
    topk_targets = targets_upper[indices]
    if len(topk_targets) < seqlen:
        topk_targets = np.pad(topk_targets, (0, seqlen - len(topk_targets)))  # torch equivalent: F.pad(topk_targets, [0, seqlen - topk_targets.size(0)])
    cumulative_dist = topk_targets.astype(float).cumsum()  # torch equivalent: topk_targets.type_as(predictions).cumsum(-1)

    gather_indices = (np.arange(0.1, 1.1, 0.1) * seqlen).astype(int) - 1  # torch equivalent: (torch.arange(0.1, 1.1, 0.1, device=device) * seqlen).type(torch.long) - 1
    binned_cumulative_dist = cumulative_dist[gather_indices]  # torch equivalent: cumulative_dist.gather(0, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).astype(float)

    pl5 = float(binned_precisions[1])
    pl2 = float(binned_precisions[4])
    pl = float(binned_precisions[9])
    auc = float(binned_precisions.mean())
    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_contact_map(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute precision scores (AUC, P@L, P@L2, P@L5) for the predicted contact map at differente ranges (local, short, medium, long).
    Reference: https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
    """
    if predictions.shape != targets.shape or predictions.size == 0:
        return {k: float("nan") for k in [
            "local_AUC", "local_P@L", "local_P@L2", "local_P@L5",
            "short_AUC", "short_P@L", "short_P@L2", "short_P@L5",
            "medium_AUC", "medium_P@L", "medium_P@L2", "medium_P@L5",
            "long_AUC", "long_P@L", "long_P@L2", "long_P@L5",
        ]}
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
    ]
    metrics: Dict[str, float] = {}
    for name, minsep, maxsep in contact_ranges:
        range_metrics = compute_contact_precision(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in range_metrics.items():
            metrics[f"{name}_{key}"] = val
    return metrics
