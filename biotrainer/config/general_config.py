import torch

from pathlib import Path
from typing import Union, List, Tuple

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol
from ..utilities import get_device


def general_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    general_category = "general"
    return ConfigKey.ROOT, [
        ConfigOption(name="device",
                     description="Select the device on which computations will be performed.",
                     category=general_category,
                     required=False,
                     default=str(get_device().type),
                     constraints=ConfigConstraints(
                         type=str,
                         allowed_values=["cpu", "mps"] + [f"cuda:{i}" for i in
                                                          range(torch.cuda.device_count())] + ["cuda"]
                         if torch.cuda.is_available() else [])),
        ConfigOption(name="interaction",
                     description="Select a method of protein-protein interaction, "
                                 "either by pair-wise multiplying or concatenating embeddings. "
                                 "Enabling it will interpret the given sequence file as an interaction dataset.",
                     category=general_category,
                     required=False,
                     constraints=ConfigConstraints(
                         type=str,
                         allowed_values=["multiply", "concat"],
                         allowed_protocols=Protocol.using_per_sequence_embeddings())),
        ConfigOption(name="seed",
                     description="Specify a seed value for all random number generators to ensure reproducibility.",
                     category=general_category,
                     required=False,
                     default=42,
                     constraints=ConfigConstraints(
                         type=int,
                     ),
                     ),
        ConfigOption(name="save_split_ids",
                     description="Choose whether to save the identifiers of data splits "
                                 "(i.e. training, validation, test) in the result file.",
                     category=general_category,
                     required=False,
                     default=False,
                     constraints=ConfigConstraints(
                         type=bool,
                     ),
                     ),
        ConfigOption(name="sanity_check",
                     description="Enable or disable sanity checks at the end of training to get immediate feedback "
                                 "about the quality of the trained model.",
                     category=general_category,
                     required=False,
                     default=True,
                     constraints=ConfigConstraints(
                         type=bool,
                     ),
                     ),
        ConfigOption(name="ignore_file_inconsistencies",
                     description="Choose whether to ignore inconsistencies in file-related configurations, "
                                 "such as labels that are not present in the sequence file.",
                     category=general_category,
                     required=False,
                     default=False,
                     constraints=ConfigConstraints(
                         type=bool,
                     ),
                     ),
        ConfigOption(name="external_writer",
                     description="Use an external writer such as tensorboard to track training progress.",
                     category=general_category,
                     required=False,
                     default="tensorboard",
                     constraints=ConfigConstraints(
                         type=str,
                         allowed_values=["tensorboard", "none"],
                     ),
                     ),
        ConfigOption(name="output_dir",
                     description="Define the directory where output files and results will be stored.",
                     category=general_category,
                     required=False,
                     default="output",
                     constraints=ConfigConstraints(
                         type=Union[str, Path],
                     ),
                     ),
        ConfigOption(name="bootstrapping_iterations",
                     description="Specify the number of bootstrap iterations for the trained model on the test set. "
                                 "Using a value of 0 disables bootstrapping.",
                     category=general_category,
                     required=False,
                     default=30,
                     constraints=ConfigConstraints(
                         type=int,
                         gte=0
                     ),
                     ),
        ConfigOption(name="validate_input",
                     description="Run checks on the given input_file data before training.",
                     category=general_category,
                     required=False,
                     default=True,
                     constraints=ConfigConstraints(
                         type=bool,
                     ),
                     )
    ]
