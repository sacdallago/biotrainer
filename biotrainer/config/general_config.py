import torch

from pathlib import Path
from typing import Union, List, Tuple

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol


def general_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    general_category = "general"
    return ConfigKey.ROOT, [
        ConfigOption(name="device",
                     type=str,
                     description="Select the device on which computations will be performed.",
                     category=general_category,
                     required=False,
                     default="cpu",
                     constraints=ConfigConstraints(
                         allowed_values=["cpu", "mps"] + [f"cuda:{i}" for i in
                                                          range(torch.cuda.device_count())] + ["cuda"]
                         if torch.cuda.is_available() else [])),
        ConfigOption(name="interaction",
                     type=str,
                     description="Select a method of protein-protein interaction, "
                                 "either by pair-wise multiplying or concatenating embeddings. "
                                 "Enabling it will interpret the given sequence file as an interaction dataset.",
                     category=general_category,
                     required=False,
                     constraints=ConfigConstraints(
                         allowed_values=["multiply", "concat"],
                         allowed_protocols=Protocol.using_per_sequence_embeddings())),
        ConfigOption(name="seed",
                     type=int,
                     description="Specify a seed value for all random number generators to ensure reproducibility.",
                     category=general_category,
                     required=False,
                     default=42),
        ConfigOption(name="save_split_ids",
                     type=bool,
                     description="Choose whether to save the identifiers of data splits "
                                 "(i.e. training, validation, test) in the result file.",
                     category=general_category,
                     required=False,
                     default=False),
        ConfigOption(name="sanity_check",
                     type=bool,
                     description="Enable or disable sanity checks at the end of training to get immediate feedback "
                                 "about the quality of the trained model.",
                     category=general_category,
                     required=False,
                     default=True),
        ConfigOption(name="ignore_file_inconsistencies",
                     type=bool,
                     description="Choose whether to ignore inconsistencies in file-related configurations, "
                                 "such as labels that are not present in the sequence file.",
                     category=general_category,
                     required=False,
                     default=False),
        ConfigOption(name="output_dir",
                     type=Union[str, Path],
                     description="Define the directory where output files and results will be stored.",
                     category=general_category,
                     required=False,
                     default="output"),
        ConfigOption(name="bootstrapping_iterations",
                     type=int,
                     description="Specify the number of bootstrap iterations for the trained model on the test set. "
                                 "Using a value of 0 disables bootstrapping.",
                     category=general_category,
                     required=False,
                     default=30,
                     constraints=ConfigConstraints(gte=0))
    ]