import logging
import numpy as np

from typing import Any, Dict, List
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold

from ..utilities import Split, DatasetSample

logger = logging.getLogger(__name__)


class CrossValidationSplitter:

    def __init__(self, protocol, cross_validation_config: Dict[str, Any]):
        self._protocol = protocol

        self._split_strategy = None
        if cross_validation_config["method"] == "hold_out":
            self._split_strategy = self.__hold_out_split

        if cross_validation_config["method"] == "k_fold":
            k = int(cross_validation_config["k"])
            if "stratified" in cross_validation_config.keys():
                stratified = eval(str(cross_validation_config["stratified"]).capitalize())
            else:
                stratified = False
            if "repeat" in cross_validation_config.keys():
                repeat = int(cross_validation_config["repeat"])
            else:
                repeat = 1
            if "nested" in cross_validation_config.keys():
                nested = eval(str(cross_validation_config["nested"]).capitalize())
            else:
                nested = False
            self._split_strategy = lambda train_dataset, val_dataset: self.__k_fold_split(k=k, stratified=stratified,
                                                                                          repeat=repeat,
                                                                                          train_dataset=train_dataset,
                                                                                          val_dataset=val_dataset)

    def split(self, train_dataset: List[DatasetSample], val_dataset: List[DatasetSample]) -> List[Split]:
        return self._split_strategy(train_dataset, val_dataset)

    @staticmethod
    def __hold_out_split(train_dataset: List[DatasetSample], val_dataset: List[DatasetSample]) -> List[Split]:
        return [Split("hold_out", train_dataset, val_dataset)]

    @staticmethod
    def __continuous_values_to_bins(ys: List) -> List:
        """
        Calculate bins for continuous target values.
        Assigns to each data_point in ys the maximum of its associated bin.
        Enables stratified k-fold cross validation for x_to_value protocols.
            params:
                ys: continuous targets
            returns: List of bins (len == len of ys)
        """
        ys_np = np.array(ys)
        maximum = np.max(ys_np)
        minimum = np.min(ys_np)

        number_of_bin_classes = 10
        bin_size = abs(maximum - minimum) / float(number_of_bin_classes)

        bins = []
        bin_dict = {}  # For logging
        bin_min = minimum
        bin_max = bin_min + bin_size
        for data_point in np.sort(ys_np):
            if bin_min <= data_point < bin_max:
                if str(bin_max) not in bin_dict.keys():
                    bin_dict[str(bin_max)] = 0

                bins.append(str(bin_max))
                bin_dict[str(bin_max)] += 1
            else:
                while data_point > bin_max:
                    bin_min += bin_size
                    bin_max += bin_size
                    if str(bin_max) not in bin_dict.keys():
                        bin_dict[str(bin_max)] = 0

                bins.append(str(bin_max))
                bin_dict[str(bin_max)] += 1

        if len(bins) != len(ys) or sum(bin_dict.values()) != len(ys):
            raise Exception(f"Could not correctly calculate bins for continuous values! "
                            f"(Number bins {len(bins)} != Number targets {len(ys)})\n"
                            f"Consider setting stratified to False or filing an issue.")

        logger.info("Transformed continuous targets to bins for Cross Validation split, statistics:")
        for idx, bin_value in enumerate(sorted(set(bins))):
            logger.info(f"{idx}: {bin_value}: {bin_dict[bin_value]} ({100 * bin_dict[bin_value] / len(bins)} %)")

        return bins

    def __k_fold_split(self, k: int, stratified: bool, repeat: int,
                       train_dataset: List[DatasetSample], val_dataset: List[DatasetSample]) -> List[Split]:
        concat_dataset = train_dataset + val_dataset
        ys = [sample.target for sample in concat_dataset]

        split_base_name = "k_fold"
        if stratified:
            logger.info(f"Splitting to stratified {k}-fold Cross Validation datasets")
            split_base_name += "-strat"
            if repeat > 1:
                split_base_name += "-rep"
                kf = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeat)
            else:
                kf = StratifiedKFold(n_splits=k)
            # Change continuous values to bins for stratified split
            if "_value" in self._protocol:
                ys = self.__continuous_values_to_bins(ys)
        else:
            logger.info(f"Splitting to {k}-fold Cross Validation datasets")
            if repeat > 1:
                split_base_name += "-rep"
                kf = RepeatedKFold(n_splits=k, n_repeats=repeat)
            else:
                kf = KFold(n_splits=k)
        all_splits = []
        for idx, (split_ids_train, split_ids_val) in enumerate(kf.split(X=concat_dataset, y=ys)):
            train_split = [concat_dataset[split_id_train] for split_id_train in split_ids_train]
            val_split = [concat_dataset[split_id_val] for split_id_val in split_ids_val]

            if repeat > 1:
                current_split_name = f"{split_base_name}-{(idx + 1) % k if (idx + 1) % k != 0 else k}-{(idx // k) + 1}"
            else:
                current_split_name = f"{split_base_name}-{idx + 1}"
            all_splits.append(Split(current_split_name, train_split, val_split))

        return all_splits
