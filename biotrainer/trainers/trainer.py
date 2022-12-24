import time
import torch
import random
import logging
import datetime

from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union, List
from torch.utils.tensorboard import SummaryWriter

from .model_factory import ModelFactory
from .TargetManager import TargetManager
from .hp_manager import HyperParameterManager
from .cv_splitter import CrossValidationSplitter
from .target_manager_utils import revert_mappings
from .embeddings import compute_embeddings, load_embeddings

from ..models import count_parameters
from ..solvers import get_solver, Solver
from ..datasets import get_collate_function, get_dataset
from ..utilities import seed_all, SanityChecker, Split, SplitResult, DatasetSample

logger = logging.getLogger(__name__)


class Trainer:
    _class_weights: Union[None, torch.FloatTensor] = None

    def __init__(self,
                 # Needed
                 sequence_file: str,
                 protocol: str, output_dir: str,
                 hp_manager: HyperParameterManager,
                 output_vars: Dict[str, Any],
                 # Optional with defaults
                 labels_file: Optional[str] = None, mask_file: Optional[str] = None,
                 embedder_name: str = "custom_embeddings",
                 embeddings_file: str = None,
                 seed: int = 42,
                 device: torch.device = None,
                 pretrained_model: str = None,
                 save_test_predictions: bool = False,
                 ignore_file_inconsistencies: bool = False,
                 limited_sample_size: int = -1,
                 cross_validation_config: Dict[str, Any] = None,
                 # Ignore rest
                 **kwargs
                 ):
        self._output_vars = output_vars

        self._sequence_file = sequence_file
        self._protocol = protocol
        self._output_dir = Path(output_dir)
        self._labels_file = labels_file
        self._mask_file = mask_file
        self._embedder_name = embedder_name
        self._embeddings_file = embeddings_file
        self._seed = seed
        self._device = device
        self._pretrained_model = pretrained_model
        self._save_test_predictions = save_test_predictions
        self._ignore_file_inconsistencies = ignore_file_inconsistencies
        self._limited_sample_size = limited_sample_size
        self._cross_validation_config = cross_validation_config
        self._cross_validation_splitter = CrossValidationSplitter(self._protocol, self._cross_validation_config)
        self._hp_manager = hp_manager

    def training_and_evaluation_routine(self):
        # 1. SETUP
        self._setup()

        # 2. EMBEDDINGS
        id2emb = self._create_and_load_embeddings()

        # 3. TARGETS => DATASETS
        target_manager = TargetManager(protocol=self._protocol, sequence_file=self._sequence_file,
                                       labels_file=self._labels_file, mask_file=self._mask_file,
                                       ignore_file_inconsistencies=self._ignore_file_inconsistencies)
        train_dataset, val_dataset, test_dataset = target_manager.get_datasets_by_annotations(id2emb)

        # 4. CLASS WEIGHTS
        self._class_weights = self._get_class_weights(target_manager=target_manager)

        # LOG COMMON VALUES FOR ALL k-fold SPLITS:
        self._output_vars['n_testing_ids'] = len(test_dataset)
        self._output_vars['n_classes'] = target_manager.number_of_outputs

        # CREATE SPLITS:
        splits = self._cross_validation_splitter.split(train_dataset=train_dataset, val_dataset=val_dataset)

        # RUN CROSS VALIDATION
        nested_cv = False
        if "nested" in self._cross_validation_config.keys():
            nested_cv = eval(str(self._cross_validation_config["nested"]).capitalize())
        hp_search_method = self._cross_validation_config["search_method"] \
            if "search_method" in self._cross_validation_config.keys() else "no_search"

        start_time_total = time.perf_counter()
        if nested_cv:
            split_results = self._run_nested_cross_validation(splits, hp_search_method)
        else:
            split_results = self._run_cross_validation(splits, hp_search_method)
        end_time_total = time.perf_counter()
        self._output_vars["elapsed_time_total"] = end_time_total - start_time_total

        best_split = self._get_best_model_of_splits(split_results)
        # 10. TESTING
        test_dataset = self._create_embeddings_dataset(test_dataset, mode="test")
        test_loader = self._create_dataloader(dataset=test_dataset, hyper_params=best_split.hyper_params)
        self._do_and_log_evaluation(best_split.solver, test_loader, target_manager)

        # 11. SANITY CHECK TODO: Think about purpose and pros and cons, flags in config, tests..
        sanity_checker = SanityChecker(output_vars=self._output_vars, mode="Warn")
        sanity_checker.check_test_results()

        return self._output_vars

    def _setup(self):
        # Seed
        seed_all(self._seed)
        # Log device
        self._output_vars['device'] = str(self._device)
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
            self._output_vars['embeddings_file'] = embeddings_file
        else:
            logger.info(f'Embeddings file was found at {embeddings_file}. Embeddings have not been computed.')
            self._embedder_name = f"precomputed_{Path(embeddings_file).stem}_{self._embedder_name}"

        # Mapping from id to embeddings
        id2emb = load_embeddings(embeddings_file_path=embeddings_file)

        # Find out feature size and add to output vars + logging
        embeddings_length = list(id2emb.values())[0].shape[-1]  # Last position in shape is always embedding length
        self._output_vars['n_features'] = embeddings_length
        logger.info(f"Number of features: {embeddings_length}")

        return id2emb

    def _get_class_weights(self, target_manager: TargetManager) -> Union[None, torch.FloatTensor]:
        # Get x_to_class specific logs and weights
        class_weights = None
        if 'class' in self._protocol or '_interaction' in self._protocol:
            self._output_vars['class_int_to_string'] = target_manager.class_int2str
            self._output_vars['class_str_to_int'] = target_manager.class_str2int
            logger.info(f"Number of classes: {self._output_vars['n_classes']}")
            # Compute class weights to pass as bias to model if option is set
            class_weights = target_manager.compute_class_weights()

        return class_weights

    def _create_embeddings_dataset(self, split: List[DatasetSample], mode: str):
        # Apply limited sample number
        if mode == "train" and self._limited_sample_size > 0:
            logger.info(f"Using limited sample size of {self._limited_sample_size} for training dataset")
            split = random.sample(split,
                                  k=min(self._limited_sample_size, len(split)))
        return get_dataset(self._protocol, split)

    def _create_dataloader(self, dataset, hyper_params: Dict) -> torch.utils.data.dataloader.DataLoader:
        # Create dataloader from dataset
        return DataLoader(
            dataset=dataset, batch_size=hyper_params["batch_size"], shuffle=hyper_params["shuffle"], drop_last=False,
            collate_fn=get_collate_function(self._protocol)
        )

    def _create_writer(self, hyper_params: Dict) -> torch.utils.tensorboard.writer.SummaryWriter:
        # Tensorboard writer
        writer = SummaryWriter(log_dir=str(self._output_dir / "runs"))

        writer.add_hparams({
            'model': hyper_params["model_choice"],
            'num_epochs': hyper_params["num_epochs"],
            'use_class_weights': hyper_params["use_class_weights"],
            'learning_rate': hyper_params["learning_rate"],
            'batch_size': hyper_params["batch_size"],
            'embedder_name': self._embedder_name,
            'seed': self._seed,
            'loss': hyper_params["loss_choice"],
            'optimizer': hyper_params["optimizer_choice"],
        }, {})

        return writer

    def _create_solver(self, split_name, model, loss_function, optimizer, writer, hyper_params: Dict) -> Solver:
        return get_solver(
            protocol=self._protocol, name=split_name,
            network=model, loss_function=loss_function, optimizer=optimizer, device=self._device,
            number_of_epochs=hyper_params["num_epochs"], patience=hyper_params["patience"],
            epsilon=hyper_params["epsilon"],
            log_writer=writer, experiment_dir=hyper_params["log_dir"],
            num_classes=self._output_vars['n_classes']
        )

    def _run_cross_validation(self, splits: List[Split], hp_search_method: str) -> List[SplitResult]:
        split_results = list()
        for split in splits:
            for hyper_params in self._hp_manager.search(mode=hp_search_method):
                logger.info(f"Training model for split {split.name}:")
                best_epoch_metrics, solver = self._do_training_by_split(outer_split=split,
                                                                        hyper_params=hyper_params)
                split_results.append(SplitResult(split.name, hyper_params, best_epoch_metrics, solver))

        return split_results

    def _run_nested_cross_validation(self, splits: List[Split], hp_search_method: str) -> List[SplitResult]:
        split_results_outer = list()
        for outer_split in splits:
            hyper_param_loss_results = list()
            for hp_iteration, hyper_params in enumerate(self._hp_manager.search(mode=hp_search_method)):
                inner_splits = self._cross_validation_splitter.nested_split(train_dataset=outer_split.train,
                                                                            hp_iteration=hp_iteration+1)
                split_results_inner = list()
                for inner_split in inner_splits:
                    logger.info(f"Training model for inner split {inner_split.name}:")
                    best_epoch_metrics_inner, s_inner = self._do_training_by_split(outer_split=outer_split,
                                                                                   hyper_params=hyper_params,
                                                                                   inner_split=inner_split)
                    split_results_inner.append(
                        SplitResult(inner_split.name, hyper_params, best_epoch_metrics_inner, s_inner))

                hyper_param_loss_results.append((hyper_params, self._get_average_loss_of_splits(split_results_inner)))

            # SELECT BEST HYPER PARAMETER COMBINATION BY AVERAGE LOSS
            hyper_param_loss_results.sort(key=lambda hp_loss_res: hp_loss_res[1],
                                          reverse=False)
            best_hyper_param_combination = hyper_param_loss_results[0][0]
            # TRAIN ON split.train, VALIDATE ON split.val
            logger.info(f"Training model for outer split {outer_split.name}:")
            best_epoch_metrics, solver = self._do_training_by_split(outer_split=outer_split,
                                                                    hyper_params=best_hyper_param_combination)
            split_results_outer.append(
                SplitResult(outer_split.name, best_hyper_param_combination, best_epoch_metrics, solver))

        return split_results_outer

    def _do_training_by_split(self, outer_split: Split, hyper_params: Dict[str, Any], inner_split: Split = None):
        split_name = inner_split.name if inner_split else outer_split.name
        train_dataset = self._create_embeddings_dataset(inner_split.train if inner_split else outer_split.train,
                                                        mode="train")
        val_dataset = self._create_embeddings_dataset(inner_split.val if inner_split else outer_split.val,
                                                      mode="val")

        if outer_split.name not in self._output_vars.keys():
            self._output_vars[outer_split.name] = {}

        if inner_split:
            self._output_vars[outer_split.name][inner_split.name] = {}
            save_dict = self._output_vars[outer_split.name][inner_split.name]
        else:
            save_dict = self._output_vars[outer_split.name]

        save_dict['n_training_ids'] = len(train_dataset)
        save_dict['n_validation_ids'] = len(val_dataset)
        save_dict['split_hyper_params'] = self._hp_manager.get_only_params_to_optimize(hyper_params)

        # 5. DATALOADERS
        train_loader = self._create_dataloader(dataset=train_dataset, hyper_params=hyper_params)
        val_loader = self._create_dataloader(dataset=val_dataset, hyper_params=hyper_params)

        # 6. MODEL, LOSS, OPTIMIZER
        # Create model, loss, optimizer
        model_factory = ModelFactory(**hyper_params)
        model, loss_function, optimizer = model_factory.create_model_loss_optimizer(
            n_classes=self._output_vars["n_classes"],
            n_features=self._output_vars["n_features"],
            class_weights=self._class_weights if hyper_params["use_class_weights"] else None)
        # Count and log number of free params
        n_free_parameters = count_parameters(model)
        save_dict['n_free_parameters'] = n_free_parameters

        # 7. WRITER
        writer = self._create_writer(hyper_params=hyper_params)

        # 8. SOLVER
        solver = self._create_solver(split_name=split_name,
                                     model=model, loss_function=loss_function, optimizer=optimizer, writer=writer,
                                     hyper_params=hyper_params)
        # if self._pretrained_model:
        #    solver.load_checkpoint(checkpoint_path=self._pretrained_model)

        # 9. TRAINING/VALIDATION
        best_epoch_metrics = self._do_and_log_training(outer_split.name, solver, train_loader, val_loader,
                                                       inner_split_name=inner_split.name if inner_split else "")

        return best_epoch_metrics, solver

    def _do_and_log_training(self, outer_split_name: str, solver, train_loader, val_loader, inner_split_name: str = ""):
        start_time_abs = datetime.datetime.now()
        start_time = time.perf_counter()
        epoch_iterations = solver.train(train_loader, val_loader)
        end_time = time.perf_counter()
        end_time_abs = datetime.datetime.now()

        # Logging
        logger.info(f'Total training time for {inner_split_name if inner_split_name != "" else outer_split_name}: '
                    f'{(end_time - start_time) / 60:.1f}[m]')

        if inner_split_name:
            save_dict = self._output_vars[outer_split_name][inner_split_name]
        else:
            save_dict = self._output_vars[outer_split_name]
        # Save training time for prosperity
        save_dict['start_time'] = start_time_abs
        save_dict['end_time'] = end_time_abs
        save_dict['elapsed_time'] = end_time - start_time

        # Save metrics from best training epoch
        save_dict['training_iteration_result_best_epoch'] = epoch_iterations[solver.get_best_epoch()]
        return epoch_iterations[solver.get_best_epoch()]

    def _get_best_model_of_splits(self, split_results: List[SplitResult]) -> SplitResult:
        split_results_sorted = sorted(split_results,
                                      key=lambda split_result: split_result.best_epoch_metrics["validation"]["loss"],
                                      reverse=False)  # Lowest to highest loss
        best_split_result = split_results_sorted[0]
        logger.info(f"Using best model from split {best_split_result.name} for test set evaluation")
        return best_split_result

    def _get_average_loss_of_splits(self, split_results: List[SplitResult]) -> float:
        sum_loss = sum([split_result.best_epoch_metrics["validation"]["loss"] for split_result in split_results])
        return sum_loss / len(split_results)

    def _do_and_log_evaluation(self, solver, test_loader, target_manager):
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        logger.info('Running final evaluation on the best model')

        solver.load_checkpoint()
        test_results = solver.inference(test_loader, calculate_test_metrics=True)

        if self._save_test_predictions:
            test_results['mapped_predictions'] = revert_mappings(protocol=self._protocol,
                                                                 test_predictions=test_results['mapped_predictions'],
                                                                 class_int2str=target_manager.class_int2str)
            self._output_vars['test_iterations_results'] = test_results
        else:
            self._output_vars['test_iterations_results'] = test_results['metrics']

        logger.info(f"Test set metrics: {test_results['metrics']}")
        logger.info(f"Extensive output information can be found at {self._output_vars['output_dir']}/out.yml")
