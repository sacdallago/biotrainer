# Global constants that do not change during the execution

from typing import Final, List  # Final is just a hint for programmers, does not actually prevent reassigning the value

# Padding and Masking
SEQUENCE_PAD_VALUE: Final[int] = 0
MASK_AND_LABELS_PAD_VALUE: Final[int] = -100

# Interaction
INTERACTION_INDICATOR: Final[str] = "&"  # seq_id1&seq_id2, e.g. SEQ1&SEQ2

# residue_to_value
RESIDUE_TO_VALUE_TARGET_DELIMITER: Final[str] = ";"

# Metrics
# Usually, a higher metric means a better model (accuracy: 0.9 > 0.8, precision: 0.5 > 0.1 ..)
# For some metrics, however, the opposite is true (loss: 0.1 > 0.2, rmse: 20.05 > 40.05)
METRICS_WITHOUT_REVERSED_SORTING: Final[List[str]] = ["loss", "mse", "rmse"]

# AAs
AMINO_ACIDS: Final[str] = "ACDEFGHIKLMNPQRSTVWXY"
