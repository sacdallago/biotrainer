from typing import List, Tuple

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol


def finetuning_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    finetuning_category = "finetuning"
    return ConfigKey.FINETUNING, [
        ConfigOption(
            name="method",
            description=f"Set the method for finetuning (currently, only lora is available)",
            category=finetuning_category,
            required=True,
            default="lora",
            constraints=ConfigConstraints(
                type=str,
                allowed_values=["lora"],
            ),
        ),
        ConfigOption(
            name="random_masking",
            description=f"If true, random masking of input sequences for masked language modeling is applied.",
            category=finetuning_category,
            required=False,
            default=False,
            constraints=ConfigConstraints(
                type=bool,
                allowed_values=[True, False],
            ),
        ),
        ConfigOption(
            name="lora_r",
            category=finetuning_category,
            description="Set the lora attention dimension (rank)",
            required=False,
            default=8,
            constraints=ConfigConstraints(
                type=int,
                gt=0
            )
        ),
        ConfigOption(
            name="lora_alpha",
            category=finetuning_category,
            description="Set the alpha parameter for lora scaling.",
            required=False,
            default=16,
            constraints=ConfigConstraints(
                type=int,
                gt=0
            )
        ),
        ConfigOption(
            name="lora_dropout",
            category=finetuning_category,
            description="Set the dropout probability for lora layers.",
            required=False,
            default=0.05,
            constraints=ConfigConstraints(
                type=float,
                gt=0,
                lt=1,
            )
        ),
        ConfigOption(
            name="lora_target_modules",
            category=finetuning_category,
            description="Set the names of the modules to apply the adapter to. "
                        "Can be a list of module names or a regex string.",
            required=False,
            default="auto",
        ),
        ConfigOption(
            name="lora_bias",
            category=finetuning_category,
            description="Set the bias type for lora. Can be 'none', 'all' or 'lora_only'.",
            required=False,
            default="none",
            constraints=ConfigConstraints(
                type=str,
                allowed_values=["none", "all", "lora_only"],
            )
        ),
    ]
