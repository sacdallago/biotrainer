from .training_factory import TrainingFactory

from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ...solvers import get_metrics_calculator
from ...utilities import get_logger, revert_mappings
from ...validations import SanityChecker, Bootstrapper

logger = get_logger(__name__)


class TestingStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.TESTING

    @staticmethod
    def _do_and_log_evaluation(context: PipelineContext, solver, test_loader, test_set_id: str):
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        solver.load_checkpoint(resume_training=False)
        test_results = solver.inference(test_loader, calculate_test_metrics=True)

        if context.config.get("save_split_ids", False):
            test_results['test_set_predictions'] = revert_mappings(protocol=context.config["protocol"],
                                                                   test_predictions=test_results['mapped_predictions'],
                                                                   class_int2str=context.target_manager.class_int2str)
            context.output_manager.add_test_set_result(test_set_id=test_set_id,
                                                       test_set_results={k: v for k, v in test_results.items()
                                                                         if k != "mapped_probabilities"})
        else:
            context.output_manager.add_test_set_result(test_set_id=test_set_id,
                                                       test_set_results={'metrics': test_results['metrics']})

        logger.info(f"Test set {test_set_id} metrics: {test_results['metrics']}")
        return test_results

    @staticmethod
    def _do_and_log_prediction(context: PipelineContext, solver, pred_loader):
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        solver.load_checkpoint(resume_training=False)
        pred_results = solver.inference(pred_loader, calculate_test_metrics=False)

        predictions = revert_mappings(protocol=context.config["protocol"],
                                      test_predictions=pred_results['mapped_predictions'],
                                      class_int2str=context.target_manager.class_int2str)
        context.output_manager.add_prediction_result(prediction_results=predictions)

        logger.info(f"Calculated predictions for {len(pred_loader)} samples!")
        return pred_results

    @staticmethod
    def _do_and_log_bootstrapping_evaluation(context: PipelineContext,
                                             metrics_calculator,
                                             test_results, test_loader, test_set_id: str):
        logger.info(f'Running bootstrapping evaluation on the best model for test set ({test_set_id})')
        bootstrapping_dict = Bootstrapper.bootstrap(protocol=context.config["protocol"],
                                                    device=context.config["device"],
                                                    bootstrapping_iterations=context.config["bootstrapping_iterations"],
                                                    metrics_calculator=metrics_calculator,
                                                    mapped_predictions=test_results["mapped_predictions"],
                                                    test_loader=test_loader)
        context.output_manager.add_test_set_result(test_set_id=test_set_id,
                                                   test_set_results={"bootstrapping": bootstrapping_dict})
        logger.info(f'Bootstrapping results for test set ({test_set_id}): {bootstrapping_dict}')

    def process(self, context: PipelineContext) -> PipelineContext:
        # TESTING
        test_datasets = context.test_datasets
        best_split = context.best_split
        assert test_datasets is not None, f"test_datasets cannot be None at the testing step!"
        assert best_split is not None, f"best_split cannot be None at the testing step!"

        finetuning = "finetuning_config" in context.config
        for test_set_id, test_dataset in test_datasets.items():
            logger.info('Running final evaluation on the best model')
            test_dataset_embeddings = TrainingFactory.create_dataset(context=context,
                                                                     split=test_dataset,
                                                                     mode="test",
                                                                     finetuning=finetuning)
            test_loader = TrainingFactory.create_dataloader(context=context, dataset=test_dataset_embeddings,
                                                            hyper_params=best_split.hyper_params, finetuning=finetuning)
            test_results = self._do_and_log_evaluation(context=context,
                                                       solver=best_split.solver,
                                                       test_loader=test_loader,
                                                       test_set_id=test_set_id)

            # ADDITIONAL EVALUATION
            metrics_calculator = get_metrics_calculator(protocol=context.config["protocol"],
                                                        device=context.config["device"],
                                                        n_classes=context.n_classes)
            # BOOTSTRAPPING
            if context.config["bootstrapping_iterations"] > 0:
                self._do_and_log_bootstrapping_evaluation(context=context,
                                                          metrics_calculator=metrics_calculator,
                                                          test_results=test_results,
                                                          test_loader=test_loader,
                                                          test_set_id=test_set_id)

            # SANITY CHECKER
            if context.config.get("sanity_check", True):
                baseline_test_dataset = context.baseline_test_datasets[test_set_id]
                baseline_test_dataset_embeddings = TrainingFactory.create_dataset(context=context,
                                                                                  split=baseline_test_dataset,
                                                                                  mode="test",
                                                                                  finetuning=False)
                baseline_test_loader = TrainingFactory.create_dataloader(context=context,
                                                                         dataset=baseline_test_dataset_embeddings,
                                                                         hyper_params=best_split.hyper_params,
                                                                         finetuning=False)
                sanity_checker = SanityChecker(training_config=context.config,
                                               n_classes=context.n_classes,
                                               n_features=context.n_features,
                                               train_dataset=context.train_dataset,
                                               val_dataset=context.val_dataset,
                                               test_dataset=baseline_test_dataset,
                                               test_loader=baseline_test_loader,
                                               metrics_calculator=metrics_calculator,
                                               test_results_dict=test_results,
                                               class_weights=context.class_weights,
                                               mode="warn")
                baseline_results, warnings = sanity_checker.check_test_results(test_set_id=test_set_id)
                if baseline_results is not None and len(baseline_results) > 0:
                    context.output_manager.add_test_set_result(test_set_id=test_set_id,
                                                               test_set_results={"test_baselines": baseline_results})
                if len(warnings) > 0:
                    context.output_manager.add_test_set_result(test_set_id=test_set_id,
                                                               test_set_results={"sanity_check_warnings": warnings})

        # PREDICTION
        prediction_dataset = context.prediction_dataset
        if prediction_dataset and len(prediction_dataset) > 0:
            logger.info(f'Calculating predictions for {len(prediction_dataset)} samples!')
            pred_dataset_embeddings = TrainingFactory.create_dataset(context=context,
                                                                     split=prediction_dataset, mode="pred")
            pred_loader = TrainingFactory.create_dataloader(context=context, dataset=pred_dataset_embeddings,
                                                            hyper_params=best_split.hyper_params, finetuning=finetuning)

            _ = self._do_and_log_prediction(context=context,
                                            solver=best_split.solver,
                                            pred_loader=pred_loader
                                            )

        return context
