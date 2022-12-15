import time
import torch
import logging
import datetime

from pathlib import Path
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union, List
from torch.utils.tensorboard import SummaryWriter

from .model_factory import ModelFactory
from .TargetManager import TargetManager
from .cv_splitter import CrossValidationSplitter
from .target_manager_utils import revert_mappings
from .embeddings import compute_embeddings, load_embeddings

from ..models import count_parameters
from ..solvers import get_solver, Solver
from ..datasets import get_collate_function, get_dataset
from ..utilities import seed_all, SanityChecker, SplitResult

logger = logging.getLogger(__name__)
output_vars = dict()


class Trainer:
    def __init__(self,
                 # Needed
                 sequence_file: str,
                 # Defined previously
                 protocol: str, output_dir: str, log_dir: str, model_factory: ModelFactory,
                 # Optional with defaults
                 labels_file: Optional[str] = None, mask_file: Optional[str] = None,
                 num_epochs: int = 200,
                 use_class_weights: bool = False,
                 batch_size: int = 128, embedder_name: str = "custom_embeddings",
                 embeddings_file: str = None,
                 shuffle: bool = True, seed: int = 42,
                 patience: int = 10, epsilon: float = 0.001,
                 device: torch.device = None,
                 pretrained_model: str = None,
                 save_test_predictions: bool = False,
                 ignore_file_inconsistencies: bool = False,
                 limited_sample_size: int = -1,
                 cross_validation_configuration: Dict[str, Any] = None,
                 # Everything else
                 **kwargs
                 ):
        if not cross_validation_configuration:
            cross_validation_configuration = {"method": "hold_out"}

        global output_vars
        output_vars = deepcopy(locals())
        output_vars.pop('kwargs')
        output_vars.pop('self')

        self._sequence_file = sequence_file
        self._protocol = protocol
        self._output_dir = Path(output_dir)
        self._log_dir = log_dir
        self._model_factory = model_factory
        self._labels_file = labels_file
        self._mask_file = mask_file
        self._num_epochs = num_epochs
        self._use_class_weights = use_class_weights
        self._batch_size = batch_size
        self._embedder_name = embedder_name
        self._embeddings_file = embeddings_file
        self._shuffle = shuffle
        self._seed = seed
        self._patience = patience
        self._epsilon = epsilon
        self._device = device
        self._pretrained_model = pretrained_model
        self._save_test_predictions = save_test_predictions
        self._ignore_file_inconsistencies = ignore_file_inconsistencies
        self._limited_sample_size = limited_sample_size
        self._cross_validation_configuration = cross_validation_configuration

    def training_and_evaluation_routine(self):
        # 1. SETUP
        self._setup()

        # 2. EMBEDDINGS
        id2emb = self._create_and_load_embeddings()

        # 3. TARGETS => DATASETS
        target_manager = TargetManager(protocol=self._protocol, sequence_file=self._sequence_file,
                                       labels_file=self._labels_file, mask_file=self._mask_file,
                                       ignore_file_inconsistencies=self._ignore_file_inconsistencies,
                                       limited_sample_size=self._limited_sample_size)
        train_dataset, val_dataset, test_dataset = target_manager.get_datasets_by_annotations(id2emb)

        # COMMON FOR ALL k-fold SPLITS:
        output_vars['n_testing_ids'] = len(test_dataset)
        output_vars['n_classes'] = target_manager.number_of_outputs
        test_dataset = get_dataset(self._protocol, test_dataset)
        test_loader = self._create_dataloader(dataset=test_dataset)

        # CREATE SPLITS:
        cross_validation_splitter = CrossValidationSplitter(self._protocol,
                                                            self._cross_validation_configuration)
        splits = cross_validation_splitter.split(train_dataset=train_dataset, val_dataset=val_dataset)

        # RUN CROSS VALIDATION
        split_results = list()
        for split in splits:
            best_epoch_metrics, solver = self._do_training_by_split(split_name=split.name,
                                                                    train_dataset=get_dataset(self._protocol,
                                                                                              split.train),
                                                                    val_dataset=get_dataset(self._protocol,
                                                                                            split.val))
            split_results.append(SplitResult(split.name, best_epoch_metrics, solver))

        solver_of_best_model = self._get_best_model_of_splits(split_results)
        # 10. TESTING
        self._do_and_log_evaluation(solver_of_best_model, test_loader, target_manager)

        # 11. SANITY CHECK TODO: Think about purpose and pros and cons, flags in config, tests..
        sanity_checker = SanityChecker(output_vars=output_vars, mode="Warn")
        sanity_checker.check_test_results()

        return output_vars

    def _setup(self):
        # Seed
        seed_all(self._seed)
        # Log device
        output_vars['device'] = str(self._device)
        logger.info(f"Using device: {self._device}")

    def _create_and_load_embeddings(self) -> Dict[str, Any]:
        # Generate embeddings if necessary, otherwise use existing embeddings and overwrite embedder_name
        embeddings_file = self._embeddings_file
        if not embeddings_file or not Path(embeddings_file).is_file():
            embeddings_file = compute_embeddings(
                embedder_name=self._embedder_name, sequence_file=self._sequence_file,
                protocol=self._protocol, output_dir=self._output_dir
            )
            # Add to out config
            output_vars['embeddings_file'] = embeddings_file
        else:
            logger.info(f'Embeddings file was found at {embeddings_file}. Embeddings have not been computed.')
            self._embedder_name = f"precomputed_{Path(embeddings_file).stem}_{self._embedder_name}"

        # Mapping from id to embeddings
        id2emb = load_embeddings(embeddings_file_path=embeddings_file)

        # Find out feature size and add to output vars + logging
        embeddings_length = list(id2emb.values())[0].shape[-1]  # Last position in shape is always embedding length
        output_vars['n_features'] = embeddings_length
        logger.info(f"Number of features: {embeddings_length}")

        return id2emb

    def _get_class_weights(self, target_manager: TargetManager) -> Union[None, torch.FloatTensor]:
        # Get x_to_class specific logs and weights
        class_weights = None
        if 'class' in self._protocol or '_interaction' in self._protocol:
            output_vars['class_int_to_string'] = target_manager.class_int2str
            output_vars['class_str_to_int'] = target_manager.class_str2int
            logger.info(f"Number of classes: {output_vars['n_classes']}")
            # Compute class weights to pass as bias to model if option is set
            if self._use_class_weights:
                class_weights = target_manager.compute_class_weights()

        return class_weights

    def _create_dataloader(self, dataset) -> torch.utils.data.dataloader.DataLoader:
        # Create dataloader from dataset
        return DataLoader(
            dataset=dataset, batch_size=self._batch_size, shuffle=self._shuffle, drop_last=False,
            collate_fn=get_collate_function(self._protocol)
        )

    def _create_writer(self) -> torch.utils.tensorboard.writer.SummaryWriter:
        # Tensorboard writer
        writer = SummaryWriter(log_dir=str(self._output_dir / "runs"))

        writer.add_hparams({
            'model': self._model_factory.model_choice,
            'num_epochs': self._num_epochs,
            'use_class_weights': self._use_class_weights,
            'learning_rate': self._model_factory.learning_rate,
            'batch_size': self._batch_size,
            'embedder_name': self._embedder_name,
            'seed': self._seed,
            'loss': self._model_factory.loss_choice,
            'optimizer': self._model_factory.optimizer_choice,
        }, {})

        return writer

    def _create_solver(self, split_name, model, loss_function, optimizer, writer) -> Solver:
        return get_solver(
            protocol=self._protocol, name=split_name,
            network=model, loss_function=loss_function, optimizer=optimizer, device=self._device,
            number_of_epochs=self._num_epochs, patience=self._patience, epsilon=self._epsilon,
            log_writer=writer, experiment_dir=self._log_dir,
            num_classes=output_vars['n_classes']
        )

    def _do_training_by_split(self, split_name: str, train_dataset, val_dataset):
        output_vars[split_name] = {}
        output_vars[split_name]['n_training_ids'] = len(train_dataset)
        output_vars[split_name]['n_validation_ids'] = len(val_dataset)

        # 4. CLASS WEIGHTS
        # TODO class_weights = self._get_class_weights(target_manager=target_manager)

        # 5. DATALOADERS
        train_loader = self._create_dataloader(dataset=train_dataset)
        val_loader = self._create_dataloader(dataset=val_dataset)

        # 6. MODEL, LOSS, OPTIMIZER
        # Create model, loss, optimizer
        model, loss_function, optimizer = self._model_factory.create_model_loss_optimizer(
            n_classes=output_vars["n_classes"],
            n_features=output_vars["n_features"],
            class_weights=None)  # TODO
        # Count and log number of free params
        n_free_parameters = count_parameters(model)
        output_vars[split_name]['n_free_parameters'] = n_free_parameters

        # 7. WRITER
        writer = self._create_writer()

        # 8. SOLVER
        solver = self._create_solver(split_name=split_name,
                                     model=model, loss_function=loss_function, optimizer=optimizer, writer=writer)
        # if self._pretrained_model:
        #    solver.load_checkpoint(checkpoint_path=self._pretrained_model)

        # 9. TRAINING/VALIDATION
        best_epoch_metrics = self._do_and_log_training(split_name, solver, train_loader, val_loader)

        return best_epoch_metrics, solver

    @staticmethod
    def _do_and_log_training(split_name, solver, train_loader, val_loader):
        start_time_abs = datetime.datetime.now()
        start_time = time.perf_counter()
        epoch_iterations = solver.train(train_loader, val_loader)
        end_time = time.perf_counter()
        end_time_abs = datetime.datetime.now()

        # Logging
        logger.info(f'Total training time for {split_name}: {(end_time - start_time) / 60:.1f}[m]')
        # TODO: ADD OVERALL TIME!

        # Save training time for prosperity
        output_vars[split_name]['start_time'] = start_time_abs
        output_vars[split_name]['end_time'] = end_time_abs
        output_vars[split_name]['elapsed_time'] = end_time - start_time

        # Save metrics from best training epoch
        output_vars[split_name]['training_iteration_result_best_epoch'] = epoch_iterations[solver.get_best_epoch()]
        return epoch_iterations[solver.get_best_epoch()]

    def _get_best_model_of_splits(self, split_results: List[SplitResult]) -> Solver:
        split_results_sorted = sorted(split_results,
                                      key=lambda split_result: split_result.best_epoch_metrics["validation"]["loss"],
                                      reverse=False)  # Lowest to highest loss
        best_split_result = split_results_sorted[0]
        logger.info(f"Using best model from split {best_split_result.name} for test set evaluation")
        return best_split_result.solver

    def _do_and_log_evaluation(self, solver, test_loader, target_manager):
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        logger.info('Running final evaluation on the best model')

        solver.load_checkpoint()
        test_results = solver.inference(test_loader, calculate_test_metrics=True)

        if self._save_test_predictions:
            test_results['mapped_predictions'] = revert_mappings(protocol=self._protocol,
                                                                 test_predictions=test_results['mapped_predictions'],
                                                                 class_int2str=target_manager.class_int2str)
            output_vars['test_iterations_results'] = test_results
        else:
            output_vars['test_iterations_results'] = test_results['metrics']

        logger.info(f"Test set metrics: {test_results['metrics']}")
        logger.info(f"Extensive output information can be found at {output_vars['output_dir']}/out.yml")
