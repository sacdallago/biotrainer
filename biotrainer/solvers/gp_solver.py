import math
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List, Any

import torch
from torch.utils.data import DataLoader

from .solver import Solver
from ..models.gp import GPModelAdapter
from ..utilities import get_logger, EpochMetrics, BiotrainerSequencePrediction, BiotrainerResiduePrediction

logger = get_logger(__name__)

try:
    import gpytorch
except Exception as e:  # pragma: no cover - optional dependency
    gpytorch = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class GPSolver(Solver):
    """
    Solver for Exact GPyTorch models wrapped by GPModelAdapter.

    Unlike neural networks, Exact GPs require the entire training set at once.
    This solver accumulates data from the dataloader before training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if gpytorch is None:  # pragma: no cover
            raise ImportError(
                "gpytorch is required for GPSolver but could not be imported."
            ) from _IMPORT_ERROR

        if not isinstance(self.network, GPModelAdapter):
            raise TypeError("GPSolver requires network to be a GPModelAdapter")

        self.mll: Optional[gpytorch.mlls._MarginalLogLikelihood] = None

    def train(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> List[EpochMetrics]:
        """
        Override base train() to accumulate full dataset before GP initialization.
        """
        # Step 1: Accumulate ALL training data
        logger.info("Accumulating training data for Exact GP...")
        train_X_list, train_y_list = [], []
        for _, X, y, lengths in training_dataloader:
            train_X_list.append(X.to(self.device))
            train_y_list.append(y.to(self.device))

        train_X = torch.cat(train_X_list, dim=0)
        train_y = torch.cat(train_y_list, dim=0)
        logger.info(f"Accumulated {train_X.shape[0]} training samples")

        # Step 2: Initialize GP with full dataset
        self.network.ensure_initialized(train_x=train_X, train_y=train_y, device=self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.network.likelihood, self.network.gp
        )

        # Step 3: Initialize optimizer
        lr = 0.1
        self.optimizer = torch.optim.AdamW(self.network.gp.parameters(), lr=lr)

        # Step 4: Training loop (on FULL dataset, not batches)
        epoch_iterations: List[EpochMetrics] = []
        self._min_loss = math.inf

        if self.start_epoch == 0:
            self.save_checkpoint(0)

        for epoch in range(self.start_epoch, self.number_of_epochs):
            # === VALIDATION ===
            val_results = self.inference(validation_dataloader, calculate_test_metrics=True)
            validation_epoch_metrics = val_results['metrics']

            # === TRAINING ===
            self.network.gp.train()
            self.network.likelihood.train()
            self.optimizer.zero_grad()

            # Forward pass on ENTIRE training set
            output = self.network.gp(train_X)
            targets = self.network._transformed_targets if self.network._is_classification else train_y
            loss = -self.mll(output, targets)
            if self.network._is_classification:
                loss = loss.sum()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute training metrics
            self.network.gp.eval()
            self.network.likelihood.eval()
            with torch.no_grad():
                _, probs_or_means = self.network.gp_outputs(train_X)
                if self.network._is_classification:
                    probabilities = probs_or_means
                    _, predicted = torch.max(probabilities, dim=1)
                else:
                    probabilities = probs_or_means.float().flatten()
                    predicted = probabilities
                train_metrics = self.metrics_calculator.compute_metrics(predicted=predicted, labels=train_y)

            # Aggregate epoch metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                training={'loss': loss.item(), **train_metrics},
                validation=validation_epoch_metrics
            )
            epoch_iterations.append(epoch_metrics)

            # Logging
            logger.info(f"Epoch {epoch}")
            logger.info(f"Training results")
            for key in epoch_metrics.training:
                logger.info(f"\t{key}: {epoch_metrics.training[key]:.2f}")
            logger.info(f"Validation results")
            for key in epoch_metrics.validation:
                logger.info(f"\t{key}: {epoch_metrics.validation[key]:.2f}")

            if self.output_manager:
                self.output_manager.add_training_iteration(split_name=self.split_name, epoch_metrics=epoch_metrics)

            # Early stopping
            if self._early_stop(current_loss=epoch_metrics.validation['loss'],
                                epoch=epoch):
                logger.info(f"Early stopping triggered - patience limit is reached! (Best epoch: {self._best_epoch})")
                return epoch_iterations

        return epoch_iterations

    def inference(self, dataloader: DataLoader, calculate_test_metrics: bool = False) -> \
            Dict[str, Union[List[Any], Dict[str, Union[float, int]]]]:
        """
        Override inference to handle GP predictions properly.
        """
        self.network.gp.eval()
        self.network.likelihood.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []
        mapped_predictions = {}
        mapped_probabilities = {}

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for seq_ids, X, y, lengths in dataloader:
                X = X.to(self.device)

                _, probs_or_means = self.network.gp_outputs(X)

                if self.network._is_classification:
                    probabilities = probs_or_means
                    _, predicted = torch.max(probabilities, dim=1)
                else:
                    probabilities = probs_or_means.float().flatten()
                    predicted = probabilities

                # Store for metrics
                all_predictions.append(predicted)
                all_labels.append(y.to(self.device))
                all_probabilities.append(probabilities)

                # Map to sequence IDs
                for idx, pred in enumerate(predicted.cpu().tolist()):
                    mapped_predictions[seq_ids[idx]] = pred
                for idx, prob in enumerate(probabilities.cpu().tolist()):
                    mapped_probabilities[seq_ids[idx]] = prob

        metrics = None
        if calculate_test_metrics:
            all_predictions_cat = torch.cat(all_predictions)
            all_labels_cat = torch.cat(all_labels)

            # Compute validation loss properly
            # TODO Loss
            """
            if self.network._is_classification:
                # For classification: use negative log predictive probability
                # This is a placeholder - you'd need proper Dirichlet evaluation
                loss = torch.nn.functional.cross_entropy(
                    torch.stack([torch.tensor(p) for p in all_probabilities]).to(self.device),
                    all_labels_cat.long()
                ).item()
            else:
                # For regression: use negative log predictive density
                # Accumulate predictive distributions
                X_all = torch.cat([X.to(self.device) for _, X, _, _ in dataloader], dim=0)

                with gpytorch.settings.fast_pred_var():
                    f_dist = self.network.gp(X_all)
                    # Get predictive distribution (includes likelihood noise)
                    pred_dist = self.network.likelihood(f_dist)

                    # Negative log predictive density
                    log_prob = pred_dist.log_prob(all_labels_cat)
                    loss = -log_prob.mean().item()  # Average NLPD
            """
            metrics = {
                **self.metrics_calculator.compute_metrics(predicted=all_predictions_cat, labels=all_labels_cat)
            }
            # TODO Negative accuracy as proxy for validation loss in classification tasks (lower = better)
            metrics['loss'] = metrics['mse'] if 'mse' in metrics else -1 * metrics.get('accuracy', 0)
            return {
            'metrics': metrics,
            'mapped_predictions': mapped_predictions,
            'mapped_probabilities': mapped_probabilities
        }

    def model_has_dropout(self):
        return True  # TODO Hack for uncertainty estimation

    def inference_monte_carlo_dropout(self, dataloader: DataLoader,
                                      n_forward_passes: int = 30,
                                      confidence_level: float = 0.05) -> List[
        Union[BiotrainerSequencePrediction, BiotrainerResiduePrediction]]:
        """
        Calculate inference results with uncertainty estimates from GP predictive distributions.

        Unlike neural networks with dropout, GPs provide analytic uncertainty through their
        predictive variance (regression) or by sampling from the predictive distribution (classification).

        Args:
            dataloader: Dataloader with embeddings
            n_forward_passes: For classification, number of samples from predictive distribution.
                             For regression, this parameter is ignored (analytic uncertainty used).
            confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
        """
        if not 0 < confidence_level < 1:
            raise ValueError(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

        if self.network.gp is None or self.network.likelihood is None:
            raise RuntimeError("GP not initialized — perform at least one training step.")

        self.network.gp.eval()
        self.network.likelihood.eval()
        predictions = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for seq_ids, X, y, lengths in dataloader:
                X = X.to(self.device)

                if self.network._is_classification:
                    # Classification: sample from Dirichlet predictive distribution
                    gp_samples = self._sample_gp_classification(X, n_forward_passes)

                    # Shape: (n_samples, batch_size, n_classes)
                    samples_tensor = torch.stack([s["probabilities"] for s in gp_samples], dim=0)

                    probs_tensor = torch.softmax(samples_tensor, dim=-1)

                    # Move to (B, S, C)
                    probs_tensor = probs_tensor.transpose(0, 1)

                    # Compute statistics across samples
                    gp_mean = probs_tensor.mean(dim=1)  # (batch_size, n_classes)
                    gp_std = probs_tensor.std(dim=1)  # (batch_size, n_classes)

                    # Confidence bounds (percentile-based)
                    lower_percentile = confidence_level / 2
                    upper_percentile = 1 - (confidence_level / 2)
                    lower_bound = probs_tensor.quantile(lower_percentile, dim=1)
                    upper_bound = probs_tensor.quantile(upper_percentile, dim=1)

                    # BALD Score
                    bald_scores = self._calculate_bald_score(probs_tensor)  # (batch_size,)

                    # Prediction by mean
                    _, prediction_by_mean = torch.max(gp_mean, dim=1)

                    for idx, prediction in enumerate(prediction_by_mean):
                        predictions.append(BiotrainerSequencePrediction(
                            seq_id=seq_ids[idx],
                            prediction=prediction.item(),
                            mcd_predictions=[s["prediction"][idx] for s in gp_samples],
                            mcd_mean=gp_mean[idx].tolist(),
                            mcd_std=gp_std[idx].tolist(),
                            mcd_lower_bound=lower_bound[idx].tolist(),
                            mcd_upper_bound=upper_bound[idx].tolist(),
                            bald_score=bald_scores[idx].item(),
                        ))

                else:
                    # Regression: use analytic GP uncertainty
                    latent_dist = self.network.gp(X)
                    pred_dist = self.network.likelihood(latent_dist)

                    # Get mean and variance
                    pred_mean = pred_dist.mean.cpu()  # (batch_size,)
                    pred_variance = pred_dist.variance.cpu()  # (batch_size,)
                    pred_std = pred_variance.sqrt()

                    # Confidence intervals using normal distribution
                    from scipy import stats
                    z_score = stats.norm.ppf(1 - confidence_level / 2)
                    lower_bound = pred_mean - z_score * pred_std
                    upper_bound = pred_mean + z_score * pred_std

                    # Generate "samples" for compatibility (just perturb mean by std)
                    # This mimics MCD output format but uses GP's analytic uncertainty
                    gp_samples_list = []
                    for _ in range(n_forward_passes):
                        # Sample from N(mean, variance)
                        sample = torch.normal(pred_mean, pred_std)
                        gp_samples_list.append(sample)

                    for idx in range(X.shape[0]):
                        predictions.append(BiotrainerSequencePrediction(
                            seq_id=seq_ids[idx],
                            prediction=pred_mean[idx].item(),
                            mcd_predictions=[s[idx].item() for s in gp_samples_list],
                            mcd_mean=pred_mean[idx].item(),  # Scalar for regression
                            mcd_std=pred_std[idx].item(),
                            mcd_lower_bound=lower_bound[idx].item(),
                            mcd_upper_bound=upper_bound[idx].item()
                        ))

        return predictions

    def _sample_gp_classification(self, X: torch.Tensor, n_samples: int) -> List[Dict]:
        """
        Sample from the GP's predictive distribution for classification.

        For Dirichlet classification, this samples from the posterior Dirichlet distribution.

        Args:
            X: Input features (batch_size, n_features)
            n_samples: Number of samples to draw

        Returns:
            List of dicts with 'prediction' and 'probabilities' keys
        """
        samples = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for _ in range(n_samples):
                # Get predictive distribution
                latent_dist = self.network.gp(X)
                pred_dist = self.network.likelihood(latent_dist)

                # For DirichletClassificationLikelihood, pred_dist is a Dirichlet
                # Sample from it to get probability vectors
                sampled_probs = pred_dist.sample()  # Shape: (batch_size, n_classes) or (n_classes, batch_size)

                # Handle potential batch_shape transpose
                if sampled_probs.shape[0] == self.network.n_classes and sampled_probs.dim() == 2:
                    sampled_probs = sampled_probs.transpose(0, 1)  # (batch_size, n_classes)

                # Get class predictions from sampled probabilities
                _, predicted = torch.max(sampled_probs, dim=-1)

                samples.append({
                    'prediction': predicted.cpu().tolist(),
                    'probabilities': sampled_probs.cpu()
                })

        return samples

    def load_checkpoint(self, checkpoint_path: Path = None, resume_training: bool = False,
                        disable_pytorch_compile: bool = True):
        return  # TODO
        if checkpoint_path:
            checkpoint_file = checkpoint_path
            self.checkpoint_type = checkpoint_path.suffix
        elif self.log_dir:
            checkpoint_file = Path(self.log_dir) / self.checkpoint_name
        else:
            checkpoint_file = Path(self._tempdir.name) / self.checkpoint_name

        if checkpoint_file.suffix != '.safetensors':
            raise Exception("Cannot load pt checkpoint because torch_pt_loading is not allowed since v1.0.0!")

        from safetensors.torch import load_file
        state = load_file(str(checkpoint_file))
        epoch = int(state['epoch'].item())

        if self.network.gp is None or self.network.likelihood is None:
            raise RuntimeError("Load GP checkpoint after the GP has been initialized (ensure_initialized).")

        # Reconstruct GP state_dict
        gp_state = {k.replace('gp.', '', 1): v for k, v in state.items() if k.startswith('gp.')}
        self.network.gp.load_state_dict(gp_state)

        # Reconstruct likelihood state_dict
        likelihood_state = {k.replace('likelihood.', '', 1): v for k, v in state.items() if k.startswith('likelihood.')}
        self.network.likelihood.load_state_dict(likelihood_state)

        if resume_training:
            self.start_epoch = epoch + 1
            if self.start_epoch == self.number_of_epochs:
                raise Exception(
                    f"Cannot resume training of checkpoint because model has already been trained for maximum number_of_epochs ({self.number_of_epochs})!"
                )
        else:
            self.start_epoch = epoch

        self.network = self.network.to(self.device)
        logger.info(f"Loaded GP model from epoch: {epoch}")

    # Export -------------------------------------------------------------------------------------------
    def save_as_onnx(self, embedding_dimension: int, output_dir: Optional[str] = None) -> str:  # pragma: no cover
        raise NotImplementedError("ONNX export is not supported for GPyTorch ExactGP models.")
