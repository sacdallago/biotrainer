from typing import Dict, Any, Tuple, List

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol


def get_default_cross_validation_config() -> Dict[str, Any]:
    return {"cross_validation_config": {"method": "hold_out"}}


def cross_validation_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    cv_category = "cross_validation"
    return ConfigKey.CROSS_VALIDATION, [
        ConfigOption(
            name="method",
            description="Define the method to use for cross-validation.",
            category=cv_category,
            required=True,
            default="hold_out",
            constraints=ConfigConstraints(
                type=str,
                allowed_values=["hold_out", "k_fold", "leave_p_out"]
            )
        ),
        ConfigOption(
            name="k",
            description="Define the number of folds to use for k-fold cross-validation.",
            category=cv_category,
            required=False,
            default=5,
            add_if=lambda config: config["method"] == "k_fold",
            constraints=ConfigConstraints(
                type=int,
                gte=2,
            )
        ),
        ConfigOption(
            name="stratified",
            description="Define whether to use stratified cross-validation, that tries to balance the classes "
                        "within the folds.",
            category=cv_category,
            required=False,
            default=False,
            add_if=lambda config: config["method"] == "k_fold",
            constraints=ConfigConstraints(
                type=bool,
            )
        ),
        ConfigOption(
            name="repeat",
            description="Define the number of times to repeat each fold. Repeating the cross-validation process "
                        "multiple times can lead to more robust performance estimates.",
            category=cv_category,
            required=False,
            default=1,
            add_if=lambda config: config["method"] == "k_fold",
            constraints=ConfigConstraints(
                type=int,
                gte=1
            )
        ),
        ConfigOption(
            name="nested",
            description="Define whether to use nested cross-validation for hyperparameter optimization.",
            category=cv_category,
            required=False,
            default=False,
            add_if=lambda config: config["method"] == "k_fold",
            constraints=ConfigConstraints(
                type=bool,
            )
        ),
        ConfigOption(
            name="nested_k",
            description="Define the number of folds to use for nested hyperparameter optimization.",
            category=cv_category,
            required=False,
            constraints=ConfigConstraints(
                type=int,
                gte=2
            )
        ),
        ConfigOption(
            name="search_method",
            description="Define the search method to use for nested cross-validation hyperparameter optimization.",
            category=cv_category,
            required=False,
            constraints=ConfigConstraints(
                type=str,
                allowed_values=["random_search", "grid_search"],
            )
        ),
        ConfigOption(
            name="n_max_evaluations_random",
            description="Limit the number of hyperparameter combinations to evaluate during "
                        "random search hyperparameter optimization.",
            category=cv_category,
            required=False,
            constraints=ConfigConstraints(
                type=int,
                gte=2
            )
        ),
        ConfigOption(
            name="choose_by",
            description="Specify which evaluation metric to use when selecting the best model "
                        "during cross validation.",
            category=cv_category,
            required=False,
            default="loss",
            constraints=ConfigConstraints(
                type=str,
                allowed_values=["loss", "accuracy", "precision", "recall"]  # TODO Add all metrics
            )
        ),
        ConfigOption(
            name="p",
            description="Define the number of samples to leave out in each iteration of leave-p-out cross-validation.",
            category=cv_category,
            required=False,
            default=5,
            add_if=lambda config: config["method"] == "leave_p_out",
            constraints=ConfigConstraints(
                type=int,
                gte=1
            )
        )
    ]
