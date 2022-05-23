import math
import torch
import logging

from pathlib import Path
from itertools import chain
from ..utilities import get_device
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Union
from .MetricsCalculator import MetricsCalculator
from contextlib import nullcontext as _nullcontext

logger = logging.getLogger(__name__)


class Solver(ABC):

    def __init__(self,
                 # Necessary
                 network, optimizer, loss_function,
                 log_writer, metrics_calculator: MetricsCalculator,
                 # Optional with defaults
                 experiment_dir: str = "",
                 number_of_epochs: int = 1000, patience: int = 20, epsilon: float = 0.001,
                 device: Union[None, str, torch.device] = None):

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.log_writer = log_writer
        self.metrics_calculator = metrics_calculator
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

    def load_checkpoint(self, checkpoint_path: str = None):
        if checkpoint_path:
            state = torch.load(checkpoint_path)
        elif self.experiment_dir:
            state = torch.load(str(Path(self.experiment_dir) / "checkpoint.pt"))
        else:
            state = torch.load(str(Path(self._tempdir.name) / "checkpoint.pt"))

        self.network.load_state_dict(state['state_dict'])
        logger.info(f"Loaded model from epoch: {state['epoch']}")
        return self.network, state['epoch']

    def save_checkpoint(self, epoch: int):
        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.experiment_dir:
            torch.save(state, str(Path(self.experiment_dir) / "checkpoint.pt"))
        else:
            torch.save(state, str(Path(self._tempdir.name) / "checkpoint.pt"))

    def early_stop(self, current_loss: float, epoch: int):
        if current_loss < (self._min_loss - self.epsilon):
            self._min_loss = current_loss
            self._stop_count = self.patience

            # Save best model (overwrite if necessary)
            self.save_checkpoint(epoch)
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
    def _aggregate_iteration_results(iteration_results):
        metrics = dict()
        iteration_metrics = [metric for metric in iteration_results[0].keys() if metric != "prediction"]
        for metric in iteration_metrics:
            metrics[metric] = sum([i[metric] for i in iteration_results]) / len(iteration_results)
        return metrics

    def train(self, training_dataloader, validation_dataloader):
        # Get things ready
        self.network = self.network.train()
        self._min_loss = math.inf

        epoch_iterations = list()

        for epoch in range(self.number_of_epochs):

            # Evaluating before testing: This way val_loss > train_loss holds for most epochs
            # If we would train before validating, the validation would benefit from the knowledge gained during
            # training, thus most likely val_loss < train_loss would be true for most epochs (and a bit confusing)
            validation_iterations = list()
            for i, (_, X, y) in enumerate(validation_dataloader):
                iteration_result = self.classification_iteration(
                    X, y, context=torch.no_grad,
                    step=len(epoch_iterations) * len(validation_dataloader) + len(validation_iterations) + 1
                )
                validation_iterations.append(iteration_result)

            train_iterations = list()
            for i, (_, X, y) in enumerate(training_dataloader):
                iteration_result = self.classification_iteration(
                    X, y,
                    step=len(epoch_iterations) * len(training_dataloader) + len(train_iterations) + 1
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

            if self.early_stop(current_loss=epoch_metrics['validation']['loss'], epoch=epoch):
                logger.info(f"Early stopping triggered!")
                return epoch_iterations

        return epoch_iterations

    def inference(self, dataloader):
        self.network = self.network.eval()

        predict_iterations = list()

        for i, (_, X, y) in enumerate(dataloader):
            iteration_result = self.classification_iteration(X, y, context=torch.no_grad)
            predict_iterations.append(iteration_result)

        return {
            'metrics': Solver._aggregate_iteration_results(predict_iterations),
            'predictions': list(chain(*[p['prediction'] for p in predict_iterations]))
        }

    @abstractmethod
    def _transform_prediction_output(self, prediction):
        """
        Parameters
        ----------
        prediction: Prediction of which shape must be transformed
        - R2C: (Batch_size x protein_Length x Number_classes)
        - S2C: (B x N)
        Returns
        -------
        - R2C: (B x N x L)
        - S2C: (B x N)
        """
        raise NotImplementedError

    def classification_iteration(self, x, y, step=1, context: Optional[Callable] = None):
        do_loss_propagation = False

        if not context:
            context = _nullcontext
            do_loss_propagation = True

        with context():
            if do_loss_propagation:
                self.optimizer.zero_grad()

            prediction = self.network(x.to(self.device))
            prediction = self._transform_prediction_output(prediction)
            prediction_probabilities = torch.softmax(prediction, dim=1)
            _, predicted_classes = torch.max(prediction_probabilities, dim=1)

            # Compute metrics
            loss = self.loss_function(prediction, y.to(self.device))
            metrics = self.metrics_calculator.calculate_metrics(y, predicted_classes)
            metrics.update({'loss': loss.item()})

            if do_loss_propagation:
                # Do a forward pass & update weights
                self.optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if self.log_writer:
                    self.log_writer.add_scalars("Step/train", metrics, step)

            metrics.update({'prediction': predicted_classes.tolist()})
            return metrics
