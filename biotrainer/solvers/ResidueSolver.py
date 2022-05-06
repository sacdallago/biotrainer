import torch

from .Solver import Solver
from sklearn import metrics


class ResidueSolver(Solver):

    def _transform_prediction_output(self, prediction):
        network_type = type(self.network).__name__
        if network_type in ["FNN", "LogReg"] :
            return prediction.permute(0, 2, 1)  # (Batch_size x protein_Length x Number_classes) => (B x N x L)

        return prediction  # No changes

    def _calculate_accuracy(self, y, predicted_classes):
        # Flatten and compute numbers for later use
        flat_y = y.flatten()
        total_y = len(y)

        # TODO: The value of the mask should be optionable!!!!
        total_to_consider = int(torch.sum(y == -100))
        flat_predicted_classes = predicted_classes.flatten().cpu()

        # Count how many match
        unmasked_accuracy = metrics.accuracy_score(flat_y, flat_predicted_classes, normalize=False)
        accuracy = (unmasked_accuracy - total_to_consider) / (total_y - total_to_consider)
        return accuracy
