import torch
from typing import Dict, List

from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from . import Bootstrapper
from ..losses import get_loss
from ..models import get_model
from ..optimizers import get_optimizer
from ..solvers import Solver, MetricsCalculator, get_solver
from ..protocols import Protocol
from ..utilities import DatasetSample, INTERACTION_INDICATOR, get_logger

logger = get_logger(__name__)


class SanityChecker:
    def __init__(self, output_vars: Dict,
                 train_val_dataset: List[DatasetSample],
                 test_dataset: List[DatasetSample],
                 test_loader: DataLoader,
                 metrics_calculator: MetricsCalculator,
                 mode: str = "warn"):
        self.output_vars = output_vars
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.test_loader = test_loader
        self.metrics_calculator = metrics_calculator
        self.mode = mode

    def _handle_result(self, result: str):
        if self.mode == "warn":
            logger.warning(result)
        elif self.mode == "error":
            #  Might be useful for integration tests later
            raise SanityException(result)

    def check_test_results(self):
        logger.info("Running sanity checks on test results..")

        self._check_metrics()
        self._check_predictions()
        self._check_baselines()

        logger.info("Sanity check on test results finished!")

    def _check_metrics(self):
        test_results = self.output_vars['test_iterations_results']

        if self.output_vars["protocol"] in Protocol.classification_protocols():
            if "metrics" in test_results.keys():
                test_result_metrics = test_results['metrics']
            else:
                self._handle_result(f"No test result metrics found!")
                return
            # Multi-class metrics
            if self.output_vars['n_classes'] > 2:
                pass
            else:
                # Binary metrics
                accuracy = test_result_metrics['accuracy']
                precision = test_result_metrics['precision']
                recall = test_result_metrics['recall']
                if accuracy == precision == recall:
                    self._handle_result(f"Accuracy ({accuracy}) == Precision == Recall for binary prediction!")

    def _check_predictions(self):
        test_results = self.output_vars['test_iterations_results']

        if "mapped_predictions" in test_results:
            predictions = list(test_results['mapped_predictions'].values())
            # Check if the model is only predicting the same value for all test samples:
            if all(prediction == predictions[0] for prediction in predictions):
                self._handle_result(f"Model is only predicting {predictions[0]} for all test samples!")

    def _check_baselines(self):
        self.output_vars["test_iterations_results"]["test_baselines"] = {}
        baseline_dict = self.output_vars["test_iterations_results"]["test_baselines"]
        baseline_dict["random_model"] = self._random_model_initialization_baseline()

        if self.output_vars["protocol"] in Protocol.classification_protocols():
            # Only for binary classification tasks at the moment:
            if self.output_vars['n_classes'] <= 2:
                baseline_dict["one_only"] = self._one_only_baseline()
                baseline_dict["zero_only"] = self._zero_only_baseline()

                if "interaction" in self.output_vars:
                    baseline_dict["bias_predictions"] = self._bias_interaction_baseline()

        elif self.output_vars["protocol"] in Protocol.regression_protocols():
            baseline_dict["mean_only"] = self._mean_only_baseline()

    def _one_only_baseline(self):
        """
        Predicts "1" for every sample in the test set. (Only for binary classification)
        """
        protocol: Protocol = self.output_vars["protocol"]
        if protocol in Protocol.per_sequence_protocols() and protocol in Protocol.classification_protocols():
            ones = torch.ones(len(self.test_dataset))
            one_only_metrics = self.metrics_calculator.compute_metrics(predicted=ones,
                                                                       labels=torch.tensor(
                                                                           [sample.target for sample in
                                                                            self.test_dataset]))
            logger.info(f"One-Only Baseline: {one_only_metrics}")
            return one_only_metrics

    def _zero_only_baseline(self):
        """
        Predicts "0" for every sample in the test set. (Only for binary classification)
        """
        protocol: Protocol = self.output_vars["protocol"]
        if protocol in Protocol.per_sequence_protocols() and protocol in Protocol.classification_protocols():
            zeros = torch.zeros(len(self.test_dataset))
            zero_only_metrics = self.metrics_calculator.compute_metrics(predicted=zeros,
                                                                        labels=torch.tensor(
                                                                            [sample.target for sample in
                                                                             self.test_dataset]))
            logger.info(f"Zero-Only Baseline: {zero_only_metrics}")
            return zero_only_metrics

    def _mean_only_baseline(self):
        """
        Predicts the mean of the test set for every sample in the test set. (Only for regression)
        """
        if self.output_vars["protocol"] in Protocol.regression_protocols():
            test_set_targets = torch.tensor([sample.target for sample in self.test_dataset])
            test_set_means = torch.full((len(test_set_targets),), torch.mean(test_set_targets).item())
            mean_only_metrics = self.metrics_calculator.compute_metrics(predicted=test_set_means,
                                                                        labels=test_set_targets)
            logger.info(f"Mean-Only Baseline: {mean_only_metrics}")
            return mean_only_metrics

    def _random_model_initialization_baseline(self):
        model = get_model(**self.output_vars)
        optimizer = get_optimizer(model_parameters=model.parameters(), **self.output_vars)
        loss = get_loss(**self.output_vars)
        solver = get_solver(name="random_init_model", network=model,
                            optimizer=optimizer, loss_function=loss,
                            num_classes=self.output_vars['n_classes'], **self.output_vars)
        random_init_inference = solver.inference(dataloader=self.test_loader, calculate_test_metrics=True)
        random_init_bootstrapping = Bootstrapper.bootstrap(protocol=self.output_vars['protocol'],
                                                           device=self.output_vars['device'],
                                                           bootstrapping_iterations=self.output_vars['bootstrapping_iterations'],
                                                           metrics_calculator=solver.metrics_calculator,
                                                           inference_results=random_init_inference,
                                                           test_loader=self.test_loader)
        logger.info(f"Random-Model Baseline: {random_init_bootstrapping}")
        return random_init_bootstrapping

    def _bias_interaction_baseline(self):
        """
            Calculates a dataset bias baseline for interactions (see for example
            Park, Marcotte 2011: https://doi.org/10.1093/bioinformatics/btr514).
            At first, it is counted how often a protein (id) is found in the positive and negative sets.
            Then, the dataset bias can be determined by calculating pearson-r for (positive_counts vs. negative counts).

            The predictor itself just sums up the positive and negative counts for both interactors and "predicts"
            the higher value.
        """
        protocol: Protocol = self.output_vars["protocol"]
        if protocol in Protocol.per_sequence_protocols() and protocol in Protocol.classification_protocols():
            # 1. Calculate protein counts
            positive_counts = {}
            negative_counts = {}

            for sample in (self.train_val_dataset + self.test_dataset):
                interactor1 = sample.seq_id.split(INTERACTION_INDICATOR)[0]
                interactor2 = sample.seq_id.split(INTERACTION_INDICATOR)[1]
                for count_dict in [positive_counts, negative_counts]:
                    if interactor1 not in count_dict:
                        count_dict[interactor1] = 0
                    if interactor2 not in count_dict:
                        count_dict[interactor2] = 0

                if sample.target == 1:
                    positive_counts[interactor1] += 1
                    positive_counts[interactor2] += 1
                else:
                    negative_counts[interactor1] += 1
                    negative_counts[interactor2] += 1

            # 2. Calculate dataset bias
            dataset_bias = pearsonr(list(positive_counts.values()), list(negative_counts.values()))

            def bias_predictor(interactor1, interactor2):
                pos_occurrences_total = positive_counts[interactor1]
                pos_occurrences_total += positive_counts[interactor2]
                neg_occurrences_total = negative_counts[interactor1]
                neg_occurrences_total += negative_counts[interactor2]

                return 0 if neg_occurrences_total >= pos_occurrences_total else 1

            # 3. Predict all test_samples from bias
            predictions = []
            test_set_targets = []
            for test_sample in self.test_dataset:
                interactor1 = test_sample.seq_id.split(INTERACTION_INDICATOR)[0]
                interactor2 = test_sample.seq_id.split(INTERACTION_INDICATOR)[1]
                predictions.append(bias_predictor(interactor1, interactor2))
                test_set_targets.append(test_sample.target)

            # 4. Calculate metrics for bias predictions
            bias_metrics = self.metrics_calculator.compute_metrics(predicted=torch.tensor(predictions),
                                                                   labels=torch.tensor(test_set_targets))
            logger.info(f"Bias Baseline for interactions: {bias_metrics}")
            logger.info(f"Dataset bias for interactions: {dataset_bias}")
            return {"dataset_bias": {"bias": float(dataset_bias.statistic), "pvalue": float(dataset_bias.pvalue)},
                    **bias_metrics}


class SanityException(Exception):
    pass
