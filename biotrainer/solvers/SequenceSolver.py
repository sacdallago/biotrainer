from .Solver import Solver
from sklearn import metrics


class SequenceSolver(Solver):

    def _transform_prediction_output(self, prediction):
        return prediction  # No changes

    def _calculate_accuracy(self, y, predicted_classes):
        return metrics.accuracy_score(y, predicted_classes, normalize=True)
