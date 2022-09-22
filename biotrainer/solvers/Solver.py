import math
import torch
import logging

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Dict, List, Any
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import chain
from contextlib import nullcontext as _nullcontext

from torch.utils.data import DataLoader

from ..utilities import get_device

logger = logging.getLogger(__name__)


class Solver(ABC):

    def __init__(self,
                 # Necessary
                 network, optimizer, loss_function,
                 # Optional with defaults
                 log_writer: Optional = None, experiment_dir: str = "",
                 number_of_epochs: int = 1000, patience: int = 20, epsilon: float = 0.001,
                 device: Union[None, str, torch.device] = None):

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
        self._min_loss = math.inf
        self._stop_count = patience
        self._tempdir = TemporaryDirectory()

        # Device handling
        self.device = get_device(device)
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
                logger.info(f"Early stopping triggered!")
                return epoch_iterations

        return epoch_iterations

    def inference(self, dataloader: DataLoader) -> Dict[str, Union[List[Any], Dict[str, Union[float, int]]]]:
        self.network = self.network.eval()

        predict_iterations = list()

        for i, (_, X, y, lengths) in enumerate(dataloader):
            iteration_result = self._training_iteration(
                X, y, context=torch.no_grad, lengths=lengths
            )
            predict_iterations.append(iteration_result)

        return {
            'metrics': Solver._aggregate_iteration_results(predict_iterations),
            'predictions': list(chain(*[p['prediction'] for p in predict_iterations]))
        }

    def load_checkpoint(self, checkpoint_path: str = None):
        if checkpoint_path:
            state = torch.load(checkpoint_path)
        elif self.experiment_dir:
            state = torch.load(str(Path(self.experiment_dir) / "checkpoint.pt"))
        else:
            state = torch.load(str(Path(self._tempdir.name) / "checkpoint.pt"))

        try:
            self.network.load_state_dict(state['state_dict'])
            self.start_epoch = state['epoch'] + 1
            logger.info(f"Loaded model from epoch: {state['epoch']}")
        except RuntimeError as e:
            raise Exception(f"Defined model architecture does not seem to match pretrained model!") from e

    def _save_checkpoint(self, epoch: int):
        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.experiment_dir:
            torch.save(state, str(Path(self.experiment_dir) / "checkpoint.pt"))
        else:
            torch.save(state, str(Path(self._tempdir.name) / "checkpoint.pt"))

    def _early_stop(self, current_loss: float, epoch: int) -> bool:
        if current_loss < (self._min_loss - self.epsilon):
            self._min_loss = current_loss
            self._stop_count = self.patience

            # Save best model (overwrite if necessary)
            self._save_checkpoint(epoch)
            return False
        else:
            if self._stop_count == 0:
                # Reload best model
                self.load_checkpoint()
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
        :param masks: An optionable mask when dealing with variable sized input
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
