import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SanityChecker:
    def __init__(self, output_vars: Dict, mode: str = "warn"):
        self.output_vars = output_vars
        self.mode = mode

    def _handle_result(self, result: str):
        if self.mode == "warn":
            logger.warning(result)
        elif self.mode == "error":
            #  Might be useful for integration tests later
            raise SanityException(result)

    def check_test_results(self):
        test_results = self.output_vars['test_iterations_results']

        if "_class" in self.output_vars["protocol"] or "_interaction" in self.output_vars["protocol"]:
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

            if "mapped_predictions" in test_results:
                predictions = list(test_results['mapped_predictions'].values())
                # Check if the model is only predicting the same value for all test samples:
                if all(prediction == predictions[0] for prediction in predictions):
                    self._handle_result(f"Model is only predicting {predictions[0]} for all test samples!")

        logger.info("Sanity check on test results finished!")


class SanityException(Exception):
    pass

