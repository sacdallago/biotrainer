from typing import Any, Dict, List
from collections import namedtuple

Split = namedtuple("Split", "name train val")


class CrossValidationSplitter:

    def __init__(self, protocol, cross_validation_config: Dict[str, Any]):
        self._protocol = protocol

        self._split_strategy = None

        if cross_validation_config["method"] == "hold_out":
            self._split_strategy = self.__hold_out_split

    def split(self, train_dataset, val_dataset) -> List[Split]:
        return self._split_strategy(train_dataset, val_dataset)

    @staticmethod
    def __hold_out_split(train_dataset, val_dataset):
        return [Split("hold_out_split", train_dataset, val_dataset)]
