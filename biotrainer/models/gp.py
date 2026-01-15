import warnings
from typing import Optional, Tuple, Union, Iterator

import torch
from torch import nn
from torch.nn import Parameter

from .biotrainer_model import BiotrainerModel


try:
    import gpytorch
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.means import LinearMean
    from gpytorch.likelihoods import GaussianLikelihood, DirichletClassificationLikelihood
except Exception as e:  # pragma: no cover - optional dependency
    gpytorch = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class _ExactGPModel(gpytorch.models.ExactGP):  # type: ignore[misc]
    """
    Internal ExactGP wrapper used by GPModelAdapter. Uses Linear mean and RBF kernel.
    Supports optional batched shape for multi-class classification (Dirichlet).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods._GaussianLikelihoodBase,  # type: ignore[attr-defined]
        num_classes: Optional[int] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size((num_classes,)) if num_classes else torch.Size()
        self.mean_module = LinearMean(train_x.shape[-1], batch_shape=batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelAdapter(BiotrainerModel):
    """
    Biotrainer adapter for an Exact GPyTorch GP model handling both regression and classification.

    Notes
    - Per-sequence tasks: expects input of shape (B, F). No sequence length dimension.
    - Task type is inferred from n_classes: if n_classes > 1 → classification, else regression.
    - The underlying GP and likelihood are lazily initialized on the first training call (when targets are provided).
    - This module's forward returns predictive tensors (means or class probabilities) for convenience, but training
      should be orchestrated by a GP-aware solver (e.g., GPSolver) that handles the ExactMarginalLogLikelihood.
    - ONNX export and torch.compile are intentionally unsupported here.
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        *,
        learn_additional_noise: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        if gpytorch is None:  # pragma: no cover
            raise ImportError(
                "gpytorch is required for GPModelAdapter but could not be imported."
            ) from _IMPORT_ERROR

        super().__init__()
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.learn_additional_noise = bool(learn_additional_noise)

        self.device = torch.device(device) if device is not None else None

        # Lazily initialized components
        self.gp: Optional[_ExactGPModel] = None
        self.likelihood: Optional[gpytorch.likelihoods.Likelihood] = None

        # For Dirichlet classification, we keep transformed targets once available
        self._transformed_targets: Optional[torch.Tensor] = None

        # Provide a placeholder trainable parameter so model.parameters() is never empty before GP is initialized.
        # This avoids optimizer construction errors in generic factory code. GPSolver will later replace the optimizer
        # with one that optimizes GP + likelihood parameters.
        self._placeholder_param: Parameter = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    # --- Required BiotrainerModel API --------------------------------------------------------------
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Returns predictive means (regression) or class probabilities (classification).
        This uses eval mode prediction path for deterministic outputs.
        """
        if self.gp is None or self.likelihood is None:
            raise RuntimeError("GPModelAdapter is not initialized yet. Call with targets once during training, or use the GPSolver which initializes lazily.")

        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.gp(x)
            pred = self.likelihood(dist)

        # Regression: return mean (B,) or (B,1). Classification: return probabilities (B, C)
        if self._is_classification:
            # DirichletClassificationLikelihood returns Dirichlet: use mean probabilities
            # For Dirichlet, pred.mean has shape (C, B) when using batch_shape=C; transpose to (B, C)
            probs = pred.mean.transpose(0, 1) if pred.mean.dim() == 2 else pred.mean
            return probs
        else:
            mean = dist.mean
            return mean if mean.dim() == 1 else mean.squeeze(-1)

    def get_downstream_model(self):
        # For checkpointing of GP state via default solver; GPSolver will handle likelihood separately
        return self.gp if self.gp is not None else self

    def compile(self):  # pragma: no cover - explicit no-op
        # Explicitly disable torch.compile for GPyTorch ExactGP here
        warnings.warn("GPModelAdapter.compile() is a no-op; torch.compile is not supported for GPs.")

    # --- Helper API used by GPSolver ----------------------------------------------------------------
    @property
    def _is_classification(self) -> bool:
        return self.n_classes and self.n_classes > 1

    # Expose parameters for optimizer construction in generic pipeline code
    # TODO Architectural smell: this should be handled more generically in the pipeline
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.gp is not None and self.likelihood is not None:
            # Chain GP and likelihood parameters
            for p in self.gp.parameters():
                yield p
            for p in self.likelihood.parameters():
                yield p
        else:
            # Yield placeholder to avoid empty parameter list errors
            yield self._placeholder_param

    def ensure_initialized(self, train_x: torch.Tensor, train_y: torch.Tensor, device: Optional[torch.device] = None):
        """Initialize likelihood and ExactGP lazily on first use."""
        if self.gp is not None and self.likelihood is not None:
            return

        dev = device or self.device or train_x.device

        if self._is_classification:
            # DirichletClassificationLikelihood expects class labels 0..C-1 as targets to build transformed targets
            dlik = DirichletClassificationLikelihood(train_y, learn_additional_noise=self.learn_additional_noise).to(dev)
            self.likelihood = dlik
            model_targets = dlik.transformed_targets
            num_classes = int(dlik.num_classes)
            self._transformed_targets = model_targets
            self.gp = _ExactGPModel(train_x.to(dev), model_targets, dlik, num_classes=num_classes).to(dev)
        else:
            glik = GaussianLikelihood().to(dev)
            self.likelihood = glik
            self.gp = _ExactGPModel(train_x.to(dev), train_y.to(dev), glik).to(dev)

    def gp_outputs(self, x: torch.Tensor) -> Tuple[gpytorch.distributions.MultivariateNormal, torch.Tensor]:
        """Return (latent_f, predictive_means_or_probs)."""
        assert self.gp is not None and self.likelihood is not None
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent = self.gp(x)
            pred = self.likelihood(latent)
        if self._is_classification:
            probs = pred.mean.transpose(0, 1) if pred.mean.dim() == 2 else pred.mean
            return latent, probs
        else:
            means = latent.mean if latent.mean.dim() == 1 else latent.mean.squeeze(-1)
            return latent, means
