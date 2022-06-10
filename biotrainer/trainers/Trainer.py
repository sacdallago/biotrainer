import time
import torch
import logging

from pathlib import Path
from copy import deepcopy
from typing import Dict, Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .EmbeddingsLoader import EmbeddingsLoader
from .PredictionIOHandler import PredictionIOHandler

from ..losses import get_loss
from ..optimizers import get_optimizer
from ..datasets import get_collate_function
from ..utilities import seed_all, get_device
from ..utilities.config import write_config_file
from ..models import get_model, count_parameters
from ..solvers import get_solver

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self,
                 # Needed
                 embeddings_loader: EmbeddingsLoader,
                 prediction_io_handler: PredictionIOHandler,
                 protocol: str, output_dir: Path,
                 # Optional with defaults
                 model_choice: str = "CNN", num_epochs: int = 200,
                 use_class_weights: bool = False, learning_rate: float = 1e-3,
                 batch_size: int = 128, shuffle: bool = True, seed: int = 42,
                 loss_choice: str = "cross_entropy_loss", optimizer_choice: str = "adam",
                 patience: int = 10, epsilon: float = 0.001,
                 device: Union[None, str, torch.device] = None,
                 # Everything else
                 **kwargs
                 ):
        self._embeddings_loader = embeddings_loader
        self._prediction_io_handler = prediction_io_handler
        self._protocol = protocol
        self._output_dir = output_dir
        self._model_choice = model_choice
        self._num_epochs = num_epochs
        self._use_loss_weights = use_class_weights
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._loss_choice = loss_choice
        self._optimizer_choice = optimizer_choice
        self._patience = patience
        self._epsilon = epsilon
        self._device = device
        self._output_vars: Dict[str,] = deepcopy(locals())
        self._loss_weights = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._model = None
        self._loss_function = None
        self._optimizer = None
        self._solver = None

    def pipeline(self):
        # Get full embedder name for experiment
        embedder_name = self._embeddings_loader.get_embedder_name()
        experiment_name = f"{embedder_name}_{self._model_choice}"
        logger.info(f'########### Experiment: {experiment_name} ###########')

        # Set seed
        seed_all(self._seed)

        # Create log directory if it does not exist yet
        log_dir = self._output_dir / self._model_choice / embedder_name
        if not log_dir.is_dir():
            logger.info(f"Creating log-directory: {log_dir}")
            log_dir.mkdir(parents=True)
        self._output_vars['log_dir'] = str(log_dir)
        self._output_vars['output_dir'] = str(self._output_dir)

        # Get device
        self._device = get_device(self._device)
        self._output_vars['device'] = str(self._device)

        # Load embeddings
        id2emb = self._embeddings_loader.load_embeddings(self._output_vars)

        # Get datasets
        train_dataset, val_dataset, test_dataset = \
            self._prediction_io_handler.get_datasets(id2emb, self._output_vars)

        # Get loss weights
        if self._use_loss_weights:
            self._loss_weights = self._prediction_io_handler.compute_loss_weights()

        # Create dataloaders and model
        self._create_dataloaders(train_dataset, val_dataset, test_dataset)
        self._create_model_and_training_params()

        # Tensorboard writer
        writer = SummaryWriter(log_dir=str(self._output_dir / "runs"))
        writer.add_hparams({
            'model': self._model_choice,
            'num_epochs': self._num_epochs,
            'use_class_weights': self._use_loss_weights,  # TODO: Rename?
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size,
            'embedder_name': embedder_name,
            'seed': self._seed,
            'loss': self._loss_choice,
            'optimizer': self._optimizer_choice,
        }, {})

        # Create solver
        self._solver = get_solver(self._protocol,
                                  network=self._model, optimizer=self._optimizer, loss_function=self._loss_function,
                                  device=self._device, number_of_epochs=self._num_epochs,
                                  patience=self._patience, epsilon=self._epsilon, log_writer=writer,
                                  experiment_dir=log_dir
                                  )

        # Count and log number of free params
        n_free_parameters = count_parameters(self._model)
        logger.info(f'Experiment: {experiment_name}. Number of free parameters: {n_free_parameters}')
        self._output_vars['n_free_parameters'] = n_free_parameters

        # Do training and log time
        self._do_and_log_training()

        # Do evaluation and log results
        self._do_and_log_evaluation()

        # Save configuration data
        write_config_file(str(log_dir / "out.yml"), self._output_vars)

        return self._output_vars

    def _create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        self._train_loader = DataLoader(
            dataset=train_dataset, batch_size=self._batch_size, shuffle=self._shuffle, drop_last=False,
            collate_fn=get_collate_function(self._protocol)
        )
        self._val_loader = DataLoader(
            dataset=val_dataset, batch_size=self._batch_size, shuffle=self._shuffle, drop_last=False,
            collate_fn=get_collate_function(self._protocol)
        )
        self._test_loader = DataLoader(
            dataset=test_dataset, batch_size=self._batch_size, shuffle=self._shuffle, drop_last=False,
            collate_fn=get_collate_function(self._protocol)
        )

    def _create_model_and_training_params(self):
        self._model = get_model(
            protocol=self._protocol, model_choice=self._model_choice,
            n_classes=self._output_vars['n_classes'], n_features=self._output_vars['n_features']
        )
        self._loss_function = get_loss(
            protocol=self._protocol, loss_choice=self._loss_choice, weight=self._loss_weights
        )
        self._optimizer = get_optimizer(
            protocol=self._protocol, optimizer_choice=self._optimizer_choice,
            learning_rate=self._learning_rate, model_parameters=self._model.parameters()
        )

    def _do_and_log_training(self):
        start_time = time.time()
        _ = self._solver.train(self._train_loader, self._val_loader)
        end_time = time.time()
        logger.info(f'Total training time: {(end_time - start_time) / 60:.1f}[m]')
        self._output_vars['start_time'] = start_time
        self._output_vars['end_time'] = end_time
        self._output_vars['elapsed_time'] = end_time - start_time

    def _do_and_log_evaluation(self):
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        self._solver.load_checkpoint()

        logger.info('Running final evaluation on the best checkpoint.')
        test_results = self._solver.inference(self._test_loader)

        if self._protocol == 'residue_to_class':
            test_results['predictions'] = ["".join(
                [self._output_vars['class_int_to_string'][p] for p in prediction]
            ) for prediction in test_results['predictions']]

        self._output_vars['test_iterations_results'] = test_results
