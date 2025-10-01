import torch

from scipy.stats import pearsonr
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader

from .bootstrapper import Bootstrapper

from ..losses import get_loss
from ..models import get_model
from ..protocols import Protocol
from ..optimizers import get_optimizer
from ..solvers import MetricsCalculator, get_solver
from ..utilities import INTERACTION_INDICATOR, get_logger

logger = get_logger(__name__)


class SanityChecker:
    def __init__(self,
                 training_config: Dict[str, Any],
                 n_classes: int,
                 n_features: int,
                 metrics_calculator: MetricsCalculator,
                 train_dataset: List,
                 val_dataset: List,
                 test_dataset: List,
                 test_loader: DataLoader,
                 test_results_dict: Dict[str, Any],
                 class_weights: Optional[torch.tensor] = None,
                 mode: str = "warn"):
        self.training_config = training_config
        self.protocol = self.training_config.get("protocol")
        self.n_classes = n_classes
        self.n_features = n_features

        self.device = self.training_config.get("device")
        self.interaction = self.training_config.get("interaction", None)
        self.bootstrapping_iterations = self.training_config.get("bootstrapping_iterations", 0)
        if self.bootstrapping_iterations <= 0:
            self.bootstrapping_iterations = 30  # Always use bootstrapping for sanity checks with default 30 iterations
        self.metrics_calculator = metrics_calculator

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.test_loader = test_loader
        self.test_results_dict = test_results_dict

        self.class_weights = class_weights
        self.mode = mode

        self._warnings = []

    def _handle_sanity_check_warning(self, warning: str):
        self._warnings.append(warning)
        if self.mode == "warn":
            logger.warning(warning)
        elif self.mode == "error":
            #  Might be useful for integration tests later
            raise SanityException(warning)

    def check_test_results(self, test_set_id) -> (Dict[str, Any], List[str]):
        logger.info(f"Running sanity checks on test set results ({test_set_id}) ..")

        self._check_metrics()
        self._check_predictions()
        baseline_dict = self._check_baselines()

        logger.info(f"Sanity check on test results ({test_set_id}) finished!")

        return baseline_dict, self._warnings

    def _check_metrics(self):
        if self.protocol in Protocol.classification_protocols():
            if "metrics" in self.test_results_dict.keys():
                test_result_metrics = self.test_results_dict['metrics']
            else:
                self._handle_sanity_check_warning(f"No test result metrics found!")
                return
            # Multi-class metrics
            if self.n_classes > 2:
                pass
            else:
                # Binary metrics
                accuracy = test_result_metrics['accuracy']
                precision = test_result_metrics['precision']
                recall = test_result_metrics['recall']
                if accuracy == precision == recall:
                    self._handle_sanity_check_warning(f"Accuracy ({accuracy}) == Precision == Recall for binary prediction!")

    def _check_predictions(self):
        mapped_predictions = self.test_results_dict.get('mapped_predictions', None)

        if mapped_predictions:
            predictions = list(mapped_predictions.values())
            # Check if the model is only predicting the same value for all test samples:
            if all(prediction == predictions[0] for prediction in predictions):
                self._handle_sanity_check_warning(f"Model is only predicting {predictions[0]} for all test samples!")

    def _check_baselines(self) -> Dict[str, Any]:
        baseline_dict = {"random_model": self._random_model_initialization_baseline()}

        if self.protocol in Protocol.classification_protocols():
            # Only for binary classification tasks at the moment:
            if self.n_classes <= 2:
                baseline_dict["one_only"] = self._one_only_baseline()
                baseline_dict["zero_only"] = self._zero_only_baseline()

                if self.interaction:
                    baseline_dict["bias_predictions"] = self._bias_interaction_baseline()

        # TODO residue_to_value
        elif self.protocol in Protocol.regression_protocols() and not self.protocol == Protocol.residue_to_value:
            baseline_dict["mean_only"] = self._mean_only_baseline()

        return baseline_dict

    def _one_only_baseline(self):
        """
        Predicts "1" for every sample in the test set. (Only for binary classification)
        """
        if self.protocol in Protocol.per_sequence_protocols() and self.protocol in Protocol.classification_protocols():
            one_only_baseline = self._value_only_baseline(value=1)
            logger.info(f"One-Only Baseline: {one_only_baseline}")
            return one_only_baseline

    def _zero_only_baseline(self):
        """
        Predicts "0" for every sample in the test set. (Only for binary classification)
        """
        if self.protocol in Protocol.per_sequence_protocols() and self.protocol in Protocol.classification_protocols():
            zero_only_baseline = self._value_only_baseline(value=0)
            logger.info(f"Zero-Only Baseline: {zero_only_baseline}")
            return zero_only_baseline

    def _mean_only_baseline(self):
        """
        Predicts the mean of the train set for every sample in the test set. (Only for regression)
        """
        if self.protocol in Protocol.regression_protocols():
            train_set_targets = torch.tensor([sample.target for sample in self.train_dataset])
            train_set_mean = torch.mean(train_set_targets).item()
            mean_only_baseline = self._value_only_baseline(value=train_set_mean)
            logger.info(f"Mean-Only Baseline: {mean_only_baseline}")
            return mean_only_baseline

    def _value_only_baseline(self, value: float):
        test_set_value_predictions = {sample.seq_id: value for sample in self.test_dataset}
        value_only_baseline = Bootstrapper.bootstrap(protocol=self.protocol,
                                                     device=self.device,
                                                     bootstrapping_iterations=self.bootstrapping_iterations,
                                                     metrics_calculator=self.metrics_calculator,
                                                     mapped_predictions=test_set_value_predictions,
                                                     test_loader=self.test_loader)
        return value_only_baseline

    def _random_model_initialization_baseline(self):
        model = get_model(n_classes=self.n_classes, n_features=self.n_features,
                          model_weights_init="normal", **self.training_config)
        optimizer = get_optimizer(protocol=self.protocol,
                                  model_parameters=model.parameters(),
                                  learning_rate=0.01,  # Model not trained, so parameter is not relevant
                                  optimizer_choice='adam',
                                  **{})
        loss = get_loss(weight=self.class_weights, **self.training_config)
        solver = get_solver(name="random_init_model", network=model,
                            optimizer=optimizer, loss_function=loss,
                            n_classes=self.n_classes, **self.training_config)
        random_init_inference = solver.inference(dataloader=self.test_loader, calculate_test_metrics=True)
        random_init_bootstrapping = Bootstrapper.bootstrap(protocol=self.protocol,
                                                           device=self.device,
                                                           bootstrapping_iterations=self.bootstrapping_iterations,
                                                           metrics_calculator=solver.metrics_calculator,
                                                           mapped_predictions=random_init_inference[
                                                               "mapped_predictions"],
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
        if self.protocol in Protocol.per_sequence_protocols() and self.protocol in Protocol.classification_protocols():
            # 1. Calculate protein counts
            positive_counts = {}
            negative_counts = {}

            for sample in (self.train_dataset + self.val_dataset + self.test_dataset):
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

            def bias_predictor(int1, int2):
                pos_occurrences_total = positive_counts[int1]
                pos_occurrences_total += positive_counts[int2]
                neg_occurrences_total = negative_counts[int1]
                neg_occurrences_total += negative_counts[int2]

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
