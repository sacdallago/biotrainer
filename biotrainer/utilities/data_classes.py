from collections import namedtuple
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict

Split = namedtuple("Split", "name train val")
SplitResult = namedtuple("SplitResult", "name, hyper_params, best_epoch_metrics, solver")
EmbeddingDatasetSample = namedtuple("EmbeddingDatasetSample", "seq_id embedding target")
SequenceDatasetSample = namedtuple("SequenceDatasetSample", "seq_id, seq_record target")


class EpochMetrics(BaseModel):
    epoch: int = Field(description="Epoch number")
    training: Dict = Field(description="Training metrics")
    validation: Dict = Field(description="Validation metrics")

    def to_dict(self):
        return {"epoch": self.epoch, "training": self.training, "validation": self.validation}
