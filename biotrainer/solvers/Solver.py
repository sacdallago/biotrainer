import os
import math
import torch
import logging

from pathlib import Path
from scipy.stats import norm
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from contextlib import nullcontext as _nullcontext
from typing import Callable, Optional, Union, Dict, List, Any

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Solver(ABC):

    def __init__(self,
                 # Necessary
                 name, network, optimizer, loss_function,
                 # Optional with defaults
                 log_writer: Optional = None, experiment_dir: str = "",
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
        self.experiment_dir = experiment_dir

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

            train_iterations = list()
            for i, (_, X, y, lengths) in enumerate(training_dataloader):
                iteration_result = self._training_iteration(
                    X, y, step=len(epoch_iterations) * len(training_dataloader) + len(train_iterations) + 1,
                    lengths=lengths
                )
                train_iterations.append(iteration_result)

            epoch_metrics = {
                'training': Solver._aggregate_iteration_results(train_iterations),
                'validation': Solver._aggregate_iteration_results(validation_iterations),
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

        return {
            'metrics': Solver._aggregate_iteration_results(predict_iterations) if calculate_test_metrics else None,
            'mapped_predictions': mapped_predictions
        }

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

        def enable_dropout(model):
            """ Function to enable the dropout layers during test-time """
            number_dropout_layers = 0
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    number_dropout_layers += 1
            if not number_dropout_layers > 0:
                raise Exception("Trying to do monte carlo dropout inference on model without dropout!")

        mapped_predictions = dict()

        for i, (seq_ids, X, y, lengths) in enumerate(dataloader):
            dropout_iterations = list()
            for idx_forward_pass in range(n_forward_passes):
                self.network = self.network.eval()
                enable_dropout(self.network)
                dropout_iteration_result = self._prediction_iteration(x=X, lengths=lengths)
                dropout_iterations.append(dropout_iteration_result)

            dropout_logits = torch.stack([dropout_iteration["logits"] for dropout_iteration in dropout_iterations])
            dropout_std_dev, dropout_mean = torch.std_mean(dropout_logits, dim=0, unbiased=True)
            z_score = norm.ppf(q=1 - (confidence_level / 2))
            confidence_range = z_score * dropout_std_dev / (n_forward_passes ** 0.5)
            prediction_by_mean = self._logits_to_predictions(dropout_mean)

            # Create dict with seq_id: prediction
            for idx, prediction in enumerate(prediction_by_mean):
                mapped_predictions[seq_ids[idx]] = {"prediction": prediction_by_mean[idx].item(),
                                                    "mcd_mean": dropout_mean[idx].item(),
                                                    "mcd_lower_bound": (
                                                            dropout_mean[idx] - confidence_range[idx]).item(),
                                                    "mcd_upper_bound": (
                                                            dropout_mean[idx] + confidence_range[idx]).item()
                                                    }

        return {
            'mapped_predictions': mapped_predictions
        }

    def auto_resume(self, training_dataloader: DataLoader, validation_dataloader: DataLoader,
                    train_wrapper):
        available_checkpoints = [checkpoint for checkpoint in os.listdir(self.experiment_dir)
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
        elif self.experiment_dir:
            state = torch.load(str(Path(self.experiment_dir) / self.checkpoint_name),
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

        if self.experiment_dir:
            torch.save(state, str(Path(self.experiment_dir) / self.checkpoint_name))
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
    def _aggregate_iteration_results(iteration_results) -> Dict[str, Any]:
        metrics = dict()
        iteration_metrics = [metric for metric in iteration_results[0].keys() if metric != "prediction"]
        for metric in iteration_metrics:
            metrics[metric] = sum([i[metric] for i in iteration_results]) / len(iteration_results)
        return metrics

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        """
        Transform network_output shape if necessary.

        :param network_output: The network_output from the ML model employed
        :return: A torch tensor of the transformed network_output
        """

        return network_output

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        An optionable transform function which goes from logits to predictions (e.g. classes)

        :param logits: The logits from the ML model employed
        :return: A torch tensor of the transformed logits
        """
        return logits

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

            prediction = self.network(x)

            # Apply logit transformations before computing loss
            prediction = self._transform_network_output(prediction)
            loss = self.loss_function(prediction, y)

            # Discretize predictions if necessary
            prediction = self._logits_to_predictions(prediction)
            metrics = self._compute_metrics(predicted=prediction, labels=y)

            if do_loss_propagation:
                # Do a forward pass & update weights
                self.optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if self.log_writer:
                    self.log_writer.add_scalars("Step/train", metrics, step)

            prediction = prediction.tolist()

            return {
                'loss': loss.item(),
                'prediction': prediction,
                **metrics
            }

    def _prediction_iteration(self, x: torch.Tensor, lengths: Optional[torch.LongTensor] = None) -> \
            Dict[str, Union[List, torch.Tensor]]:

        with torch.no_grad():
            # Move everything on device
            x = x.to(self.device)
            prediction_logits = self.network(x)
            # Apply transformations
            prediction_logits = self._transform_network_output(prediction_logits)
            # Discretize predictions if necessary
            prediction = self._logits_to_predictions(prediction_logits)
            return {"prediction": prediction.tolist(),
                    "logits": prediction_logits}

    @abstractmethod
    def _compute_metrics(
            self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:
        """
        Computes metrics, such as accuracy or RMSE between predicted (from the model used) and labels (from the data).

        :param predicted: The predicted label/value for each sample
        :param labels: The actual label for each sample
        :return: A dictionary of metrics specific to the type of problem, e.g. accuracy for class predictions, or
                 RMSE for regression tasks.
        """
        raise NotImplementedError
