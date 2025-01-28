from typing import Dict, Any

from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol

def get_default_cross_validation_config() -> Dict[str, Any]:
    return {"cross_validation_config": {"method": "hold_out"}}

def cross_validation_config(protocol: Protocol):
    cv_category = "cross_validation"  # TODO Categories
    return "cross_validation_config", [  # TODO Keys -> Enum
        ConfigOption(
            name="method",
            type=str,
            required=True,
            default="hold_out",
            constraints=ConfigConstraints(
                allowed_values=["hold_out", "k_fold", "leave_p_out"]
            )
        ),
        ConfigOption(
            name="k",
            type=int,
            required=False,
            constraints=ConfigConstraints(
                gte=2,
            )
        ),
        ConfigOption(
            name="stratified",
            type=bool,
            required=False,
            default=False,
        ),
        ConfigOption(
            name="repeat",
            type=int,
            required=False,
            default=1,
            constraints=ConfigConstraints(
                gte=1
            )
        ),
        ConfigOption(
            name="nested",
            type=bool,
            required=False,
            default=False,
        ),
        ConfigOption(
            name="nested_k",
            type=int,
            required=False,
            constraints=ConfigConstraints(
                gte=2
            )
        ),
        ConfigOption(
            name="search_method",
            type=str,
            required=False,
            constraints=ConfigConstraints(
                allowed_values=["random_search", "grid_search"],
            )
        ),
        ConfigOption(
            name="n_max_evaluations_random",
            type=int,
            required=False,
            constraints=ConfigConstraints(
                gte=2
            )
        ),
        ConfigOption(
            name="choose_by",
            type=str,
            required=False,
            default="loss",
            constraints=ConfigConstraints(
                allowed_values=["loss", "accuracy", "precision", "recall"]  # TODO Add all metrics
            )
        ),
        ConfigOption(
            name="p",
            type=int,
            required=False,
            default=5,
            constraints=ConfigConstraints(
                gte=1
            )
        )
    ]