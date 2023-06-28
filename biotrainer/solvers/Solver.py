import os
import math
import torch
import logging
import torchmetrics

from pathlib import Path
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from torchmetrics import SpearmanCorrCoef
from contextlib import nullcontext as _nullcontext
from typing import Callable, Optional, Union, Dict, List, Any

from .solver_utils import get_mean_and_confidence_range

logger = logging.getLogger(__name__)


class Solver(ABC):

    def __init__(self,
                 # Necessary
                 name, network, optimizer, loss_function,
                 # Optional with defaults
                 log_writer: Optional = None, log_dir: str = "",
                 number_of_epochs: int = 1000, patience: int = 20, epsilon: float = 0.001,
                 device: Union[None, str, torch.device] = None,
                 # Used by classification subclasses
                 num_classes: Optional[int] = 0):

        self.checkpoint_name = f"{name}_checkpoint.pt"
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.log_writer = log_writer
        self.start_epoch = 0
        self.number_of_epochs = number_of_epochs
        self.patience = patience
        self.epsilon = epsilon
        self.log_dir = log_dir

        # Early stopping internal variables
        self._best_epoch = 0
        self._min_loss = math.inf
        self._stop_count = patience
        self._tempdir = TemporaryDirectory()

        # Device handling
        self.device = device
        self.network = network.to(self.device)

    def __del__(self):
        self._tempdir.cleanup()

    def train(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> List[Dict[str, Any]]:
        # Get things ready
        self.network = self.network.train()
        self._min_loss = math.inf

        epoch_iterations = list()

        for epoch in range(self.start_epoch, self.number_of_epochs):

            # Evaluating before testing: This way val_loss > train_loss holds for most epochs
            # If we would train before validating, the validation would benefit from the knowledge gained during
            # training, thus most likely val_loss < train_loss would be true for most epochs (and a bit confusing)
            validation_iterations = list()
            for i, (_, X, y, lengths) in enumerate(validation_dataloader):
                iteration_result = self._training_iteration(
                    X, y, step=len(epoch_iterations) * len(validation_dataloader) + len(validation_iterations) + 1,
                    context=torch.no_grad, lengths=lengths
                )
                validation_iterations.append(iteration_result)
            validation_epoch_metrics = self._compute_metrics()

            train_iterations = list()
            for i, (_, X, y, lengths) in enumerate(training_dataloader):
                iteration_result = self._training_iteration(
                    X, y, step=len(epoch_iterations) * len(training_dataloader) + len(train_iterations) + 1,
                    lengths=lengths
                )
                train_iterations.append(iteration_result)
            train_epoch_metrics = self._compute_metrics()

            epoch_metrics = {
                'training': {**Solver._aggregate_iteration_losses(train_iterations), **train_epoch_metrics},
                'validation': {**Solver._aggregate_iteration_losses(validation_iterations), **validation_epoch_metrics},
                'epoch': epoch
            }

            epoch_iterations.append(epoch_metrics)

            # Logging
            logger.info(f"Epoch {epoch}")
            logger.info(f"Training results")
            for key in epoch_metrics['training']:
                logger.info(f"\t{key}: {epoch_metrics['training'][key]:.2f}")
            logger.info(f"Validation results")
            for key in epoch_metrics['validation']:
                logger.info(f"\t{key}: {epoch_metrics['validation'][key]:.2f}")

            if self.log_writer:
                self.log_writer.add_scalars("Epoch/train", epoch_metrics['training'], epoch)
                self.log_writer.add_scalars("Epoch/validation", epoch_metrics['validation'], epoch)
                self.log_writer.add_scalars("Epoch/comparison", {
                    'training_loss': epoch_metrics['training']['loss'],
                    'validation_loss': epoch_metrics['validation']['loss'],
                }, epoch)

            if self._early_stop(current_loss=epoch_metrics['validation']['loss'], epoch=epoch):
                logger.info(f"Early stopping triggered! (Best epoch: {self._best_epoch})")
                return epoch_iterations

        return epoch_iterations

    def inference(self, dataloader: DataLoader, calculate_test_metrics: bool = False) -> \
            Dict[str, Union[List[Any], Dict[str, Union[float, int]]]]:
        self.network = self.network.eval()

        predict_iterations = list()
        mapped_predictions = dict()
        mapped_probabilities = dict()

        for i, (seq_ids, X, y, lengths) in enumerate(dataloader):
            if calculate_test_metrics:  # For test set, y must be valid targets
                iteration_result = self._training_iteration(
                    X, y, context=torch.no_grad, lengths=lengths
                )
            else:  # For new predictions, y is ignored
                iteration_result = self._prediction_iteration(x=X, lengths=lengths)

            predict_iterations.append(iteration_result)
            # Create dict with seq_id: prediction
            for idx, prediction in enumerate(iteration_result["prediction"]):
                mapped_predictions[seq_ids[idx]] = prediction

            for idx, probability in enumerate(iteration_result["probabilities"]):
                mapped_probabilities[seq_ids[idx]] = probability

        metrics = None
        if calculate_test_metrics:
            metrics = {**Solver._aggregate_iteration_losses(predict_iterations), **self._compute_metrics()}

        return {
            'metrics': metrics,
            'mapped_predictions': mapped_predictions,
            'mapped_probabilities': mapped_probabilities
        }

    @staticmethod
    def _enable_dropout(model):
        """ Function to enable the dropout layers during test-time """
        number_dropout_layers = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                if m.p > 0.0:
                    number_dropout_layers += 1
        if not number_dropout_layers > 0:
            raise Exception("Trying to do monte carlo dropout inference on model without dropout!")

    def _do_dropout_iterations(self, X, lengths, n_forward_passes):
        dropout_iterations = []
        for idx_forward_pass in range(n_forward_passes):
            self.network = self.network.eval()
            self._enable_dropout(self.network)
            dropout_iteration_result = self._prediction_iteration(x=X, lengths=lengths)
            dropout_iterations.append(dropout_iteration_result)

        return dropout_iterations

    def inference_monte_carlo_dropout(self, dataloader: DataLoader,
                                      n_forward_passes: int = 30,
                                      confidence_level: float = 0.05):
        """
        Calculate inference results from existing models for given embeddings

            dataloader: Dataloader with embeddings
            n_forward_passes: Times to repeat calculation with different dropout nodes enabled
            confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
        """
        if not 0 < confidence_level < 1:
            raise Exception(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

        mapped_predictions = {}

        for i, (seq_ids, X, y, lengths) in enumerate(dataloader):
            dropout_iterations = self._do_dropout_iterations(X, lengths, n_forward_passes)

            dropout_raw_values = torch.stack([torch.tensor(dropout_iteration["probabilities"])
                                              for dropout_iteration in dropout_iterations], dim=1)

            dropout_mean, confidence_range = get_mean_and_confidence_range(values=dropout_raw_values,
                                                                           dimension=1,
                                                                           n=n_forward_passes,
                                                                           confidence_level=confidence_level)
            prediction_by_mean = self._probabilities_to_predictions(dropout_mean)

            # Create dict with seq_id: prediction
            for idx, prediction in enumerate(prediction_by_mean):
                mapped_predictions[seq_ids[idx]] = {"prediction": prediction_by_mean[idx].item(),
                                                    "mcd_mean": dropout_mean[idx],
                                                    "mcd_lower_bound": (
                                                            dropout_mean[idx] - confidence_range[idx]),
                                                    "mcd_upper_bound": (
                                                            dropout_mean[idx] + confidence_range[idx])
                                                    }

        return {
            'mapped_predictions': mapped_predictions
        }

    def auto_resume(self, training_dataloader: DataLoader, validation_dataloader: DataLoader,
                    train_wrapper):
        available_checkpoints = [checkpoint for checkpoint in os.listdir(self.log_dir)
                                 if checkpoint.split("_")[0] == self.checkpoint_name.split("_")[0]]
        if self.checkpoint_name in available_checkpoints:
            # Hold out
            if "hold_out" in self.checkpoint_name:
                self.load_checkpoint(resume_training=True)
                return train_wrapper(self)
            # K_fold and leave_p_out: Checkpoint is already there, now check if it has been trained completely
            # Sort by name and by last split number (differs from alphabetical order because str(9) > str(10))
            available_checkpoints_sorted = sorted(available_checkpoints,
                                                  key=lambda ch_pt: (ch_pt.split("_checkpoint.pt")[0].split("-")[0:-1],
                                                                     int(ch_pt.split("_checkpoint.pt")[0].split("-")[-1]
                                                                         )
                                                                     )
                                                  )
            if available_checkpoints_sorted.index(self.checkpoint_name) < (len(available_checkpoints_sorted) - 1):
                # Do inference on train and val to get metrics of training
                self.load_checkpoint(resume_training=False)
                train_set_result = self.inference(training_dataloader, calculate_test_metrics=True)
                val_set_result = self.inference(validation_dataloader, calculate_test_metrics=True)
                return {
                    "training": train_set_result["metrics"],
                    "validation": val_set_result["metrics"],
                    "epoch": self.start_epoch
                }

        # Checkpoint not available or not sure if checkpoint training has finished
        logger.info(f"No pretrained checkpoint ({self.checkpoint_name}) found for auto_resume. "
                    f"Training new model from scratch!")
        return train_wrapper(self)

    def load_checkpoint(self, checkpoint_path: str = None, resume_training: bool = False):
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=torch.device(self.device))
        elif self.log_dir:
            state = torch.load(str(Path(self.log_dir) / self.checkpoint_name),
                               map_location=torch.device(self.device))
        else:
            state = torch.load(str(Path(self._tempdir.name) / self.checkpoint_name),
                               map_location=torch.device(self.device))

        try:
            self.network.load_state_dict(state['state_dict'])
            if resume_training:
                self.start_epoch = state['epoch'] + 1
                if self.start_epoch == self.number_of_epochs:
                    raise Exception(f"Cannot resume training of checkpoint because model has already "
                                    f"been trained for maximum number_of_epochs ({self.number_of_epochs})!")
            else:
                self.start_epoch = state['epoch']
            logger.info(f"Loaded model from epoch: {state['epoch']}")
        except RuntimeError as e:
            raise Exception(f"Defined model architecture does not seem to match pretrained model!") from e

    def get_best_epoch(self) -> int:
        if self.start_epoch > 0:
            return self._best_epoch - self.start_epoch
        else:
            return self._best_epoch

    def _save_checkpoint(self, epoch: int):
        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.log_dir:
            torch.save(state, str(Path(self.log_dir) / self.checkpoint_name))
        else:
            torch.save(state, str(Path(self._tempdir.name) / self.checkpoint_name))

    def _early_stop(self, current_loss: float, epoch: int) -> bool:
        if current_loss < (self._min_loss - self.epsilon):
            self._min_loss = current_loss
            self._stop_count = self.patience
            self._best_epoch = epoch
            # Save best model (overwrite if necessary)
            self._save_checkpoint(epoch)
            return False
        else:
            if self._stop_count == 0:
                # Trigger early stop
                return True
            else:
                self._stop_count = self._stop_count - 1
                return False

    @staticmethod
    def _aggregate_iteration_losses(iteration_results) -> Dict[str, Any]:
        mean_loss = sum([i["loss"] for i in iteration_results]) / len(iteration_results)
        return {"loss": mean_loss}

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        """
        Transform network_output shape if necessary.

        :param network_output: The network_output from the ML model employed
        :return: A torch tensor of the transformed network_output
        """

        return network_output

    def _logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        An optional transform function which goes from logits to probabilities (softmax)
        Regression tasks simply ignore this step

        :param logits: The logits from the ML model employed
        :return: A torch tensor of the transformed logits (torch.softmax)
        """
        return logits

    def _probabilities_to_predictions(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        An optional transform function which goes from logits or probabilities to predictions (e.g. classes)
        Regression tasks simply ignore this step

        :param probabilities: The probabilities from the transformed logits of the ML model employed
        :return: A torch tensor of the discretised probabilities
        """
        return probabilities

    def _training_iteration(
            self, x: torch.Tensor, y: torch.Tensor, step=1, context: Optional[Callable] = None,
            lengths: Optional[torch.LongTensor] = None
    ) -> Dict[str, Union[float, list, Dict[str, Union[float, int]]]]:
        do_loss_propagation = False

        if not context:
            context = _nullcontext
            do_loss_propagation = True

        # Move everything on device
        x = x.to(self.device)
        y = y.to(self.device)

        with context():
            if do_loss_propagation:
                self.optimizer.zero_grad()

            logits = self.network(x)

            # Apply logit transformations before computing loss
            logits = self._transform_network_output(logits)
            loss = self.loss_function(logits, y)

            # Transform logits to probabilities if necessary
            probabilities = self._logits_to_probabilities(logits)
            # Discretize predictions if necessary
            prediction = self._probabilities_to_predictions(probabilities)
            metrics = self._compute_metrics(predicted=prediction, labels=y)

            if do_loss_propagation:
                # Do a forward pass & update weights
                self.optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if self.log_writer:
                    self.log_writer.add_scalars("Step/train", metrics, step)

            return {
                'loss': loss.item(),
                'prediction': prediction.tolist(),
                'probabilities': probabilities
            }

    def _prediction_iteration(self, x: torch.Tensor, lengths: Optional[torch.LongTensor] = None) -> \
            Dict[str, Union[List, torch.Tensor]]:

        with torch.no_grad():
            # Move everything on device
            x = x.to(self.device)
            logits = self.network(x)
            # Apply transformations
            logits = self._transform_network_output(logits)
            # Transform logits to probabilities if necessary
            probabilities = self._logits_to_probabilities(logits)
            # Discretize predictions if necessary
            prediction = self._probabilities_to_predictions(probabilities)
            return {"prediction": prediction.tolist(),
                    "probabilities": probabilities.tolist()}

    @staticmethod
    def _compute_metric(metric: torchmetrics.Metric, predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Utility function to calculate metrics either on a per-epoch or a per-batch basis

        :param metric: torchmetrics object
        :param predicted: The predicted label/value for each sample
        :param labels: The actual label for each sample

        :return: metric result calculated via metric object
        """
        if predicted is None and labels is None:
            # Per epoch
            metric_result = metric.compute()
            metric.reset()
            return metric_result
        else:
            # Per batch
            if metric.__class__ == SpearmanCorrCoef:
                # SCC only accepts float tensors
                return metric(predicted.cpu().float(), labels.cpu().float())
            return metric(predicted.cpu(), labels.cpu())

    @abstractmethod
    def _compute_metrics(
            self, predicted: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[int, float]]:
        """
        Computes metrics, such as accuracy or RMSE between predicted (from the model) and labels (from the data).

        If both, predicted and labels are None, metrics for the whole epoch are calculated and the metric objects
        are reset

        :param predicted: The predicted label/value for each sample
        :param labels: The actual label for each sample
        :return: A dictionary of metrics specific to the type of problem, e.g. accuracy for class predictions, or
                 RMSE for regression tasks.
        """
        raise NotImplementedError
