from typing import Tuple, List

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol
from ..losses import get_available_losses_dict
from ..models import get_available_models_dict
from ..optimizers import get_available_optimizers_dict


def model_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    model_category = "model"
    return ConfigKey.ROOT, [
        ConfigOption(
            name="model_choice",
            description="Choose a specific model architecture from the available predefined models.",
            category=model_category,
            required=False,
            default="LightAttention" if protocol == Protocol.residues_to_class else "FNN",
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                type=str,
                allowed_values=list(get_available_models_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="optimizer_choice",
            description="Choose an optimizer from the available predefined optimizers.",
            category=model_category,
            required=False,
            default="adam",
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                type=str,
                allowed_values=list(get_available_optimizers_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="learning_rate",
            description="Define a learning rate for the optimizer.",
            category=model_category,
            required=False,
            default=1e-3,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                type=float,
                gt=0,
                lt=1
            )
        ),
        ConfigOption(
            name="dropout_rate",
            description="Define one or more dropout rates to be applied in the model (if applicable).",
            category=model_category,
            required=False,
            default=0.25,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                type=float,
                gte=0,
                lt=1
            )
        ),
        ConfigOption(
            name="epsilon",
            description="Define an epsilon value for the optimizer algorithm.",
            category=model_category,
            required=False,
            default=1e-3,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                type=float,
                gt=0,
                lt=1
            )
        ),
        ConfigOption(
            name="loss_choice",
            description="Choose an loss function from the available predefined losses.",
            category=model_category,
            required=False,
            default="mean_squared_error" if protocol in Protocol.regression_protocols() else "cross_entropy_loss",
            constraints=ConfigConstraints(
                type=str,
                allowed_values=list(get_available_losses_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="disable_pytorch_compile",
            description="Disable the automatic compilation of PyTorch models, "
                        "which can be useful for debugging or when encountering compatibility issues.",
            category=model_category,
            required=False,
            default=True,
            constraints=ConfigConstraints(
                type=bool,
            )
        )
    ]
