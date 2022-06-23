# Global constants that do not change during the execution

from typing import Final  # Final is just a hint for programmers, does not actually prevent reassigning the value

# Padding and Masking
SEQUENCE_PAD_VALUE: Final[int] = 0
MASK_AND_LABELS_PAD_VALUE: Final[int] = -100
