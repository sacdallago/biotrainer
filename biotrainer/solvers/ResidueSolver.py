from .Solver import Solver


class ResidueSolver(Solver):

    def _transform_prediction_output(self, prediction):
        network_type = type(self.network).__name__
        if network_type in ["FNN", "LogReg"] :
            return prediction.permute(0, 2, 1)  # (Batch_size x protein_Length x Number_classes) => (B x N x L)

        return prediction  # No changes
