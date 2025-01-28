import torch

from typing import Union
from pathlib import Path

from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol


def general_config(protocol: Protocol):
    return "", [
        ConfigOption(name="device",
                     type=str,
                     required=False,
                     default="cpu",
                     constraints=ConfigConstraints(
                         allowed_values=["cpu", "mps"] + [f"cuda:{i}" for i in
                                                          range(torch.cuda.device_count())] + ["cuda"]
                         if torch.cuda.is_available() else [])),
        ConfigOption(name="interaction", type=str, required=False,
                     constraints=ConfigConstraints(
                         allowed_values=["multiply", "concat"],
                         allowed_protocols=Protocol.using_per_sequence_embeddings())),
        ConfigOption(name="seed", type=int, required=False, default=42),
        ConfigOption(name="save_split_ids", type=bool, required=False, default=False),
        ConfigOption(name="sanity_check", type=bool, required=False, default=True),
        ConfigOption(name="ignore_file_inconsistencies", type=bool, required=False, default=False),
        ConfigOption(name="output_dir", type=Union[str, Path], required=False, default="output"),
        ConfigOption(name="bootstrapping_iterations", type=int, required=False, default=30,
                     constraints=ConfigConstraints(gte=0))
    ]