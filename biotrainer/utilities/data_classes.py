from collections import namedtuple

Split = namedtuple("Split", "name train val")
SplitResult = namedtuple("SplitResult", "name, hyper_params, best_epoch_metrics, solver")
DatasetSample = namedtuple("DatasetSample", "seq_id embedding target")
