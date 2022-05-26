import torch
from .Solver import Solver


class ResidueClassificationSolver(Solver):

    def _transform_prediction_output(self, prediction):
        network_type = type(self.network).__name__
        if network_type in ["FNN", "LogReg"] :
            prediction = prediction.permute(0, 2, 1)  # (Batch_size x protein_Length x Number_classes) => (B x N x L)

        prediction_probabilities = torch.softmax(prediction, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)
        return prediction, predicted_classes
