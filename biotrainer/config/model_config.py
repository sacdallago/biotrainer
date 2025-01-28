from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol
from ..losses import get_available_losses_dict
from ..models import get_available_models_dict
from ..optimizers import get_available_optimizers_dict


def model_config(protocol: Protocol):
    return "", [
        ConfigOption(
            name="model_choice",
            type=str,
            required=False,
            default="LightAttention" if protocol == Protocol.residues_to_class else "FNN",
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                allowed_values=list(get_available_models_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="optimizer_choice",
            type=str,
            required=False,
            default="adam",
            constraints=ConfigConstraints(
                allowed_values=list(get_available_optimizers_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="learning_rate",
            type=float,
            required=False,
            default=1e-3,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                gt=0,
                lt=1
            )
        ),
        ConfigOption(
            name="dropout_rate",
            type=float,
            required=False,
            default=0.25,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                gt=0,
                lt=1
            )
        ),
        ConfigOption(
            name="epsilon",
            type=float,
            required=False,
            default=1e-3,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                gt=0,
                lt=1
            )
        ),
        ConfigOption(
            name="loss_choice",
            type=str,
            required=False,
            default="mean_squared_error" if protocol in Protocol.regression_protocols() else "cross_entropy_loss",
            constraints=ConfigConstraints(
                allowed_values=list(get_available_losses_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="disable_pytorch_compile",
            type=bool,
            required=False,
            default=True
        )
    ]
