import numpy as np

from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, LeavePOut

from ..protocols import Protocol
from ..utilities import Split, get_logger

logger = get_logger(__name__)


class CrossValidationSplitter:

    def __init__(self, protocol: Protocol, cross_validation_config: Dict[str, Any]):
        self._protocol = protocol

        self._split_strategy = None
        if cross_validation_config["method"] == "hold_out":
            self._split_strategy = self._hold_out_split

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
            if "nested_k" in cross_validation_config.keys():
                nested_k = int(cross_validation_config["nested_k"])
                self._nested_split_strategy = lambda train_dataset, current_outer_k, hp_iteration: \
                    self._k_fold_split(k=nested_k, stratified=stratified, nested=True, repeat=repeat,
                                       train_dataset=train_dataset, val_dataset=[],
                                       current_outer_k=current_outer_k, hp_iteration=hp_iteration)
            self._split_strategy = lambda train_dataset, val_dataset: \
                self._k_fold_split(k=k, stratified=stratified, nested=False, repeat=repeat, train_dataset=train_dataset,
                                   val_dataset=val_dataset)

        if cross_validation_config["method"] == "leave_p_out":
            p = int(cross_validation_config["p"])
            self._split_strategy = lambda train_dataset, val_dataset: \
                self._leave_p_out_split(p=p, train_dataset=train_dataset, val_dataset=val_dataset)

    def split(self, train_dataset: List, val_dataset: List) -> List[Split]:
        return self._split_strategy(train_dataset, val_dataset)

    def nested_split(self, train_dataset: List, current_outer_k: int, hp_iteration: int) -> List[Split]:
        return self._nested_split_strategy(train_dataset, current_outer_k, hp_iteration)

    @staticmethod
    def _hold_out_split(train_dataset: List, val_dataset: List) -> List[Split]:
        return [Split("hold_out", train_dataset, val_dataset)]

    @staticmethod
    def _continuous_values_to_bins(ys: List, number_bins: Optional[int] = 10) -> List:
        """
        Calculate bins for continuous target values.
        Assigns to each data_point in ys the maximum of its associated bin.
        Enables stratified k-fold cross validation for x_to_value protocols.
            params:
                ys: continuous targets
                number_bins: number of bins to create
            returns: List of bins (len == len of ys)
        """
        ys_np = np.array(ys)
        maximum = np.max(ys_np)
        minimum = np.min(ys_np)

        bin_size = abs(maximum - minimum) / float(number_bins)

        bins = []
        bin_dict = {}  # For logging
        bin_min = minimum
        bin_max = bin_min + bin_size
        for data_point in np.sort(ys_np):
            if bin_min <= data_point <= bin_max:
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

    def _k_fold_split(self, k: int, stratified: bool, nested: bool, repeat: int,
                      train_dataset: List, val_dataset: List,
                      hp_iteration: Optional[int] = None,
                      current_outer_k: Optional[int] = None) -> List[Split]:
        concat_dataset = train_dataset + val_dataset
        ys = [sample.target for sample in concat_dataset]

        split_base_name = "k_fold"
        inner_or_outer_splits = "(inner split)" if not nested else "(outer splits)"
        if stratified:
            logger.info(f"Splitting to stratified {k}-fold Cross Validation datasets {inner_or_outer_splits}")
            split_base_name += "-strat"
            if repeat > 1:
                split_base_name += "-rep"
                kf = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeat)
            else:
                kf = StratifiedKFold(n_splits=k)
            # Change continuous values to bins for stratified split
            if self._protocol in Protocol.regression_protocols():
                ys = self._continuous_values_to_bins(ys)
        else:
            logger.info(f"Splitting to {k}-fold Cross Validation datasets {inner_or_outer_splits}")
            if repeat > 1:
                split_base_name += "-rep"
                kf = RepeatedKFold(n_splits=k, n_repeats=repeat)
            else:
                kf = KFold(n_splits=k)
        if current_outer_k:
            split_base_name += f"-{current_outer_k}"
        if hp_iteration:
            split_base_name += f"-hp-{hp_iteration}"
        if nested:
            split_base_name += "-nested"
        all_splits = []
        for idx, (split_ids_train, split_ids_val) in enumerate(kf.split(X=concat_dataset, y=ys)):
            train_split = [concat_dataset[split_id_train] for split_id_train in split_ids_train]
            val_split = [concat_dataset[split_id_val] for split_id_val in split_ids_val]

            if repeat > 1:
                # Repeated splits are labeled e.g. 1-1, 1-2, 2-1, 2-2, 3-1, 3-2 .. (for 3==2 and repeat==2)
                current_repeated_split = f"{(idx + 1) % k if (idx + 1) % k != 0 else k}-{(idx // k) + 1}"
                current_split_name = f"{split_base_name}-{current_repeated_split}"
            else:
                current_split_name = f"{split_base_name}-{idx + 1}"

            all_splits.append(Split(current_split_name, train_split, val_split))

        return all_splits

    @staticmethod
    def _leave_p_out_split(p: int,
                           train_dataset: List, val_dataset: List) -> List[Split]:
        concat_dataset = train_dataset + val_dataset

        lpo = LeavePOut(p=p)
        n_created_datasets = lpo.get_n_splits(X=concat_dataset)
        logger.info(f"Splitting to leave_{p}_out Cross Validation datasets")
        logger.info(f"Number of created datasets: {n_created_datasets}")
        if n_created_datasets > 1000:
            logger.warning(f"Number of created datasets is very high! Consider using a smaller value for p or another "
                           f"cross validation method.")

        all_splits = []
        for idx, (split_ids_train, split_ids_val) in enumerate(lpo.split(X=concat_dataset)):
            train_split = [concat_dataset[split_id_train] for split_id_train in split_ids_train]
            val_split = [concat_dataset[split_id_val] for split_id_val in split_ids_val]
            all_splits.append(Split(f"leave_{p}_out-{idx + 1}", train_split, val_split))

        return all_splits
