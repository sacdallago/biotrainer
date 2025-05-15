from dataclasses import dataclass
from collections import namedtuple

Split = namedtuple("Split", "name train val")
SplitResult = namedtuple("SplitResult", "name, hyper_params, best_epoch_metrics, solver")
DatasetSample = namedtuple("DatasetSample", "seq_id embedding target")


@dataclass
class EpochMetrics:
    epoch: int
    training: dict
    validation: dict

    def to_dict(self):
        return {"epoch": self.epoch, "training": self.training, "validation": self.validation}
