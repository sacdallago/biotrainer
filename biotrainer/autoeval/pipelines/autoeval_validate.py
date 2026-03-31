from typing import Optional

from ..core import AutoEvalFramework, AutoEvalMode
from ..autoeval_frameworks import framework_factory

from ...bioengineer import ZeroShotMethod


def validate_input(framework,
                   zero_shot_method: Optional[ZeroShotMethod],
                   min_seq_length: Optional[int],
                   max_seq_length: Optional[int]) -> AutoEvalFramework:
    framework_obj = framework_factory(framework)

    if framework_obj is None:
        raise ValueError(f"Unsupported framework: {framework}")

    is_supervised = framework_obj.get_mode() == AutoEvalMode.SUPERVISED
    if is_supervised:  # Supervised frameworks
        if zero_shot_method is not None:
            raise ValueError("Zero-shot method must not be provided for a supervised framework!")
        if min_seq_length is None or max_seq_length is None:
            raise ValueError("min_seq_length and max_seq_length must be provided for a supervised framework!")
        if min_seq_length >= max_seq_length:
            raise ValueError("min_seq_length must be less than max_seq_length")

        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be greater than 0")
    else:  # Zero-Shot frameworks
        if zero_shot_method is None:
            raise ValueError("Zero-shot method must be provided for a zero-shot framework!")

    return framework_obj
