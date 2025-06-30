from dataclasses import dataclass
from collections import namedtuple

Split = namedtuple("Split", "name train val")
SplitResult = namedtuple("SplitResult", "name, hyper_params, best_epoch_metrics, solver")
EmbeddingDatasetSample = namedtuple("EmbeddingDatasetSample", "seq_id embedding target")
SequenceDatasetSample = namedtuple("SequenceDatasetSample", "seq_id, seq_record target")

@dataclass
class EpochMetrics:
    epoch: int
    training: dict
    validation: dict

    def to_dict(self):
        return {"epoch": self.epoch, "training": self.training, "validation": self.validation}
