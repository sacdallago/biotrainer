import torch

from torchmetrics import Metric


class NDCG(Metric):
    """
    Normalized Discounted Cumulative Gain metric compatible with torchmetrics.

    This implementation follows ProteinGym's approach with:
    - Min-max normalization for gains
    - Quantile-based or absolute top-k selection
    - Filtering of zero gains

    Adapted from ProteinGym Benchmarking - Original Source:
    https://github.com/OATML-Markslab/ProteinGym/blob/37ea726885452197125f841a33320341d665bc3f/proteingym/performance_DMS_benchmarks.py

    Args:
        quantile: If True, uses top k as percentage. If False, uses absolute k.
        top: If quantile=True, percentage (e.g., 10 for top 10%). 
             If quantile=False, absolute number of positions.
             If -1: Use all positions.
    """

    def __init__(self, quantile: bool = True, top: int = 10, **kwargs):
        super().__init__(**kwargs)

        self.quantile = quantile
        self.top = top

        # State for accumulating predictions and labels across batches
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate predictions and targets."""
        self.preds.append(preds.detach().cpu())
        self.target.append(target.detach().cpu())

    def compute(self) -> torch.Tensor:
        """Compute NDCG from accumulated predictions and targets."""
        # Concatenate all batches
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        # Compute NDCG using ProteinGym implementation
        ndcg_value = self._calc_ndcg(target, preds)

        return torch.tensor(ndcg_value)

    def _calc_ndcg(self, y_true: torch.Tensor, y_score: torch.Tensor) -> float:
        """
        Calculate NDCG following ProteinGym implementation.

        Args:
            y_true: True fitness scores (higher is better)
            y_score: Predicted scores (higher is better)

        Returns:
            NDCG score (0 to 1)
        """

        # Determine k
        if self.top == -1:
            k = len(y_true)
        elif self.quantile:
            k = int(torch.floor(torch.tensor(y_true.shape[0] * (self.top / 100))))
        else:
            k = self.top

        # Normalize gains to [0, 1]
        y_min = torch.min(y_true)
        y_max = torch.max(y_true)
        gains = torch.zeros_like(y_true) if y_max == y_min else (y_true - y_min) / (y_max - y_min)

        # Get ranks based on predictions (higher score = better rank)
        ranks = torch.argsort(torch.argsort(-y_score)) + 1

        # Filter to top k positions
        mask_k = ranks <= k
        ranks_k = ranks[mask_k]
        gains_k = gains[mask_k]

        # Remove zero gains
        mask_nonzero = gains_k != 0
        ranks_fil = ranks_k[mask_nonzero]
        gains_fil = gains_k[mask_nonzero]

        # If no non-zero gains in top k, return 0
        if len(ranks_fil) == 0:
            return 0.0

        # Calculate DCG
        dcg = torch.sum(gains_fil / torch.log2(ranks_fil.float() + 1))

        # Calculate ideal DCG (based on true ranking)
        ideal_ranks = torch.argsort(torch.argsort(-gains)) + 1
        ideal_mask_k = ideal_ranks <= k
        ideal_ranks_k = ideal_ranks[ideal_mask_k]
        ideal_gains_k = gains[ideal_mask_k]
        ideal_mask_nonzero = ideal_gains_k != 0
        ideal_ranks_fil = ideal_ranks_k[ideal_mask_nonzero]
        ideal_gains_fil = ideal_gains_k[ideal_mask_nonzero]

        idcg = torch.sum(ideal_gains_fil / torch.log2(ideal_ranks_fil.float() + 1))

        # Normalize (handle edge case where idcg = 0)
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg.item()
