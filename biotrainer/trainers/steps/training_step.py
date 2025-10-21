import time
import datetime

from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Any

from .training_factory import TrainingFactory

from ..cv_splitter import CrossValidationSplitter
from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ...solvers import Solver
from ...models import count_parameters
from ...utilities import get_logger, Split, SplitResult, EpochMetrics, METRICS_WITHOUT_REVERSED_SORTING

logger = get_logger(__name__)


class TrainingStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.TRAINING

    def _run_cross_validation(self, context: PipelineContext, splits: List[Split]) -> List[SplitResult]:
        split_results = list()
        for split in splits:
            for hyper_params in context.hp_manager.search(mode="no_search"):
                logger.info(f"Training model for split {split.name}:")
                best_epoch_metrics, solver = self._do_training_by_split(context=context,
                                                                        outer_split=split,
                                                                        hyper_params=hyper_params)
                split_results.append(SplitResult(split.name, hyper_params, best_epoch_metrics, solver))

        return split_results

    def _run_nested_cross_validation(self, context: PipelineContext, cross_validation_config: Dict[str, Any],
                                     cross_validation_splitter: CrossValidationSplitter,
                                     splits: List[Split]) -> List[SplitResult]:
        hp_search_method = cross_validation_config["search_method"]
        split_results_outer = list()
        for outer_k, outer_split in enumerate(splits):
            hyper_param_metric_results = list()
            for hp_iteration, hyper_params in enumerate(context.hp_manager.search(mode=hp_search_method)):
                inner_splits = cross_validation_splitter.nested_split(train_dataset=outer_split.train,
                                                                      current_outer_k=outer_k + 1,
                                                                      hp_iteration=hp_iteration + 1
                                                                      )
                split_results_inner = list()
                for inner_split in inner_splits:
                    logger.info(f"Training model for inner split {inner_split.name}:")
                    best_epoch_metrics_inner, s_inner = self._do_training_by_split(context=context,
                                                                                   outer_split=outer_split,
                                                                                   hyper_params=hyper_params,
                                                                                   inner_split=inner_split)
                    split_results_inner.append(
                        SplitResult(inner_split.name, hyper_params, best_epoch_metrics_inner, s_inner))

                hyper_param_metric_results.append((hyper_params,
                                                   self._get_average_of_chosen_metric_for_splits(
                                                       cross_validation_config, split_results_inner)
                                                   ))

            # SELECT BEST HYPER PARAMETER COMBINATION BY AVERAGE METRIC TO OPTIMIZE
            hyper_param_metric_results = self._sort_according_to_chosen_metric(
                cross_validation_config=cross_validation_config,
                list_to_sort=hyper_param_metric_results,
                key=lambda hp_metric_res: hp_metric_res[1]
            )

            best_hyper_param_combination = hyper_param_metric_results[0][0]
            # TRAIN ON split.train, VALIDATE ON split.val
            logger.info(f"Training model for outer split {outer_split.name} with best hyper_parameter combination "
                        f"{context.hp_manager.get_only_params_to_optimize(best_hyper_param_combination)} "
                        f"(criterion: {cross_validation_config['choose_by']}):")
            best_epoch_metrics, solver = self._do_training_by_split(context=context,
                                                                    outer_split=outer_split,
                                                                    hyper_params=best_hyper_param_combination)
            split_results_outer.append(
                SplitResult(outer_split.name, best_hyper_param_combination, best_epoch_metrics, solver))

        return split_results_outer

    @staticmethod
    def _get_split_name(split: Split, inner_split: Optional[Split] = None) -> str:
        return f"{split.name}-{inner_split.name}" if inner_split else split.name

    def _do_training_by_split(self, context: PipelineContext,
                              outer_split: Split,
                              hyper_params: Dict[str, Any],
                              inner_split: Split = None):
        # Necessary for differentiation of splits during nested k-fold cross validation
        current_split = inner_split if inner_split else outer_split
        current_split_name = self._get_split_name(current_split, inner_split)

        # DATASETS
        finetuning = "finetuning_config" in context.config
        train_dataset = TrainingFactory.create_dataset(context, current_split.train, mode="train", finetuning=finetuning)
        val_dataset = TrainingFactory.create_dataset(context, current_split.val, mode="val", finetuning=finetuning)

        split_hyper_params = context.hp_manager.get_only_params_to_optimize(hyper_params)
        context.output_manager.add_split_specific_values(split_name=current_split_name,
                                                         split_specific_values={'n_training_ids': len(train_dataset),
                                                                                'n_validation_ids': len(val_dataset),
                                                                                'split_hyper_params': split_hyper_params})
        if context.config.get("save_split_ids"):
            context.output_manager.add_split_specific_values(split_name=current_split_name,
                                                             split_specific_values={
                                                                 'training_ids': [sample.seq_id for sample in
                                                                                  current_split.train],
                                                                 'validation_ids': [sample.seq_id for sample in
                                                                                    current_split.val],
                                                             })

        # DATALOADERS
        train_loader = TrainingFactory.create_dataloader(context=context, dataset=train_dataset,
                                                         hyper_params=hyper_params, finetuning=finetuning)
        val_loader = TrainingFactory.create_dataloader(context=context, dataset=val_dataset,
                                                       hyper_params=hyper_params, finetuning=finetuning)

        # MODEL, LOSS, OPTIMIZER
        model, loss_function, optimizer = TrainingFactory.create_model_loss_optimizer(context=context,
                                                                                      hyper_params=hyper_params)

        # Count and log number of free params
        n_free_parameters = count_parameters(model)
        context.output_manager.add_split_specific_values(split_name=current_split_name,
                                                         split_specific_values={
                                                             'n_free_parameters': n_free_parameters,
                                                         })

        # SOLVER
        solver = TrainingFactory.create_solver(context=context,
                                               split_name=current_split_name,
                                               model=model, loss_function=loss_function, optimizer=optimizer,
                                               hyper_params=hyper_params)
        # TRAINING/VALIDATION
        if context.config.get("auto_resume", False):
            best_epoch_metrics = solver.auto_resume(training_dataloader=train_loader, validation_dataloader=val_loader,
                                                    train_wrapper=lambda untrained_solver: self._do_and_log_training(
                                                        context=context,
                                                        split_name=current_split_name,
                                                        solver=untrained_solver,
                                                        train_loader=train_loader,
                                                        val_loader=val_loader,
                                                    )
                                                    )
        else:
            pretrained_model = context.config.get("pretrained_model")
            if pretrained_model:
                solver.load_checkpoint(checkpoint_path=pretrained_model, resume_training=True)
            best_epoch_metrics = self._do_and_log_training(context=context,
                                                           split_name=current_split_name,
                                                           solver=solver,
                                                           train_loader=train_loader,
                                                           val_loader=val_loader)

        # Save metrics from best training epoch
        context.output_manager.add_split_specific_values(split_name=current_split_name,
                                                         split_specific_values={
                                                             'best_training_epoch_metrics': best_epoch_metrics.to_dict()
                                                         })

        return best_epoch_metrics, solver

    @staticmethod
    def _do_and_log_training(context: PipelineContext, split_name: str, solver: Solver, train_loader: DataLoader,
                             val_loader: DataLoader) -> EpochMetrics:
        start_time_abs = str(datetime.datetime.now().isoformat())
        start_time = time.perf_counter()
        epoch_iterations = solver.train(train_loader, val_loader)
        end_time = time.perf_counter()
        end_time_abs = str(datetime.datetime.now().isoformat())

        # Logging
        logger.info(f'Total training time for split {split_name}: {end_time - start_time} [s]')

        # Save training time for prosperity
        context.output_manager.add_split_specific_values(split_name=split_name,
                                                         split_specific_values={'start_time': start_time_abs,
                                                                                'end_time': end_time_abs,
                                                                                'elapsed_time': end_time - start_time}
                                                         )

        return epoch_iterations[solver.get_best_epoch()]

    @staticmethod
    def _sort_according_to_chosen_metric(cross_validation_config: Dict[str, Any], list_to_sort: List, key):
        choose_by_metric = cross_validation_config["choose_by"]
        reverse = choose_by_metric not in METRICS_WITHOUT_REVERSED_SORTING
        return sorted(list_to_sort,
                      key=key,
                      reverse=reverse)

    @staticmethod
    def _get_best_model_of_splits(context: PipelineContext, cross_validation_config: Dict[str, Any],
                                  split_results: List[SplitResult]) -> SplitResult:
        choose_by_metric = cross_validation_config["choose_by"]
        split_results_sorted = TrainingStep._sort_according_to_chosen_metric(
            cross_validation_config=cross_validation_config,
            list_to_sort=split_results,
            key=lambda split_result:
            split_result.best_epoch_metrics.validation[
                choose_by_metric])
        best_split_result = split_results_sorted[0]
        if len(split_results) > 1:  # Not for hold_out cross validation
            logger.info(f"Using best model from split {best_split_result.name} "
                        f"(criterion: {choose_by_metric}) for test set evaluation")
            context.output_manager.add_derived_values({'best_split': best_split_result.name})
        return best_split_result

    @staticmethod
    def _get_average_of_chosen_metric_for_splits(cross_validation_config: Dict[str, Any],
                                                 split_results: List[SplitResult]) -> float:
        choose_by_metric = cross_validation_config['choose_by']
        sum_metric = sum([split_result.best_epoch_metrics.validation[choose_by_metric]
                          for split_result in split_results])
        return sum_metric / len(split_results)

    @staticmethod
    def _log_average_result_of_splits(context: PipelineContext, split_results: List[SplitResult]):
        n = len(split_results)
        if n > 1:  # Not for hold_out cross validation
            average_dict = {}
            result_metric_keys = split_results[0].best_epoch_metrics.validation.keys()
            for key in result_metric_keys:
                average_dict[key] = sum(
                    [split_result.best_epoch_metrics.validation[key] for split_result in split_results]) / n
            logger.info(f"Average split results: {average_dict}")
            context.output_manager.add_derived_values({'average_outer_split_results': average_dict})

    def process(self, context: PipelineContext) -> PipelineContext:
        del context.id2emb  # No longer required and should not be used later in the routine
        context.id2emb = None

        # CREATE SPLITS: (TODO Might be possible to refactor this to get all splits before training)
        cross_validation_config = context.config["cross_validation_config"]
        cross_validation_splitter = CrossValidationSplitter(context.config["protocol"],
                                                            cross_validation_config)
        splits = cross_validation_splitter.split(train_dataset=context.train_dataset,
                                                 val_dataset=context.val_dataset)

        # RUN CROSS VALIDATION
        nested_cv = False
        if "nested" in cross_validation_config.keys():
            nested_cv = eval(str(cross_validation_config["nested"]).capitalize())

        start_time_training = time.perf_counter()
        if nested_cv:
            split_results = self._run_nested_cross_validation(context=context,
                                                              cross_validation_config=cross_validation_config,
                                                              cross_validation_splitter=cross_validation_splitter,
                                                              splits=splits)
        else:
            split_results = self._run_cross_validation(context=context, splits=splits)
        end_time_training = time.perf_counter()
        training_elapsed_time = end_time_training - start_time_training

        context.output_manager.add_derived_values({'training_elapsed_time': training_elapsed_time})
        logger.info(f"Total elapsed time for training: {training_elapsed_time} [s]")
        self._log_average_result_of_splits(context, split_results)
        best_split = self._get_best_model_of_splits(context=context, cross_validation_config=cross_validation_config,
                                                    split_results=split_results)

        context.best_split = best_split
        return context
