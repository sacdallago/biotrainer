from .Solver import Solver


class SequenceSolver(Solver):

    def _transform_prediction_output(self, prediction):
        return prediction  # No changes
