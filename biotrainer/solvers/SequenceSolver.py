from .Solver import Solver
from sklearn import metrics


class SequenceSolver(Solver):

    def _transform_prediction_output(self, prediction):
        return prediction  # No changes

    def _calculate_accuracy(self, y, predicted_classes):
        return metrics.accuracy_score(y.cpu(), predicted_classes.cpu(), normalize=True)
