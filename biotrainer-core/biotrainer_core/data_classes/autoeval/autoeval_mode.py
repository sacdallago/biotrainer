from enum import Enum


class AutoEvalMode(Enum):
    SUPERVISED = "SUPERVISED"
    UNSUPERVISED = "UNSUPERVISED"  # E(mbedding)A(nnotation)T(ransfer)
    ZERO_SHOT = "ZERO_SHOT"
    ZERO_SHOT_CONTACT = "ZERO_SHOT_CONTACT"
    SUPERVISED_CONTACT_ATTENTION = "SUPERVISED_CONTACT_ATTENTION"


DEV_MODE_INDICATOR = "§dev"
DEV_MODE_ABLATED_INDICATOR = "§ablated"  # Can only exist after full evaluation

class AutoEvalDevMode(Enum):
    # Plain train-val-test split, evaluated in the pipeline regardless of dev_mode.
    TRAIN_VAL_TEST = "TRAIN_VAL_TEST"
    # Subsampled test development dataset (zeroshot, contact) -
    # full evaluation contains dev but can also be evaluated on the disjunct set
    DEV_DATASET_TEST = "DEV_DATASET_TEST"
    # Subsampled training development dataset (unsupervised) -
    # no substraction or ablation of the dev vs. eval performance possible
    DEV_DATASET_TRAIN = "DEV_DATASET_TRAIN"
