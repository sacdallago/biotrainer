import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SanityChecker:
    def __init__(self, output_vars: Dict, mode: str = "Warn"):
        self.output_vars = output_vars
        self.mode = mode

    def handle_result(self, result: str):
        if self.mode == "Warn":
            logger.warning(result)
        elif self.mode == "Error":
            raise SanityException(result)

    def check_test_results(self):
        test_results = self.output_vars['test_iterations_results']

        if "_class" in self.output_vars["protocol"]:
            if "metrics" in test_results.keys():
                test_result_metrics = test_results['metrics']
            else:
                test_result_metrics = test_results
            # Multi-class metrics
            if self.output_vars['n_classes'] > 2:
                pass
            else:
                # Binary metrics
                accuracy = test_result_metrics['accuracy']
                precision = test_result_metrics['precision']
                recall = test_result_metrics['recall']
                if accuracy == precision == recall:
                    self.handle_result(f"Accuracy ({accuracy} == Precision == Recall for binary prediction!")

            if "predictions" in test_results:
                predictions = test_results['predictions']
                # Check if the model is only predicting the same value for all test samples:
                if all(prediction == predictions[0] for prediction in predictions):
                    self.handle_result(f"Model is only predicting {predictions[0]} for all test samples!")

        logger.info("Sanity check on test results successful!")


class SanityException(Exception):
    pass

