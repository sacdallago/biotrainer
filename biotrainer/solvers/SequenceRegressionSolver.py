import torch
from .Solver import Solver


class SequenceRegressionSolver(Solver):

    def _transform_prediction_output(self, prediction):
        """
        Flattens prediction tensor (BxN) to (B) and makes sure it uses float values
        """
        return prediction.flatten().float(), prediction.flatten().float()
