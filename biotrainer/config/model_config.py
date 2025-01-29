from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol
from ..losses import get_available_losses_dict
from ..models import get_available_models_dict
from ..optimizers import get_available_optimizers_dict


def model_config(protocol: Protocol):
    model_category = "model"
    return "", [
        ConfigOption(
            name="model_choice",
            type=str,
            description="Choose a specific model architecture from the available predefined models.",
            category=model_category,
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
            description="Choose an optimizer from the available predefined optimizers.",
            category=model_category,
            required=False,
            default="adam",
            constraints=ConfigConstraints(
                allowed_values=list(get_available_optimizers_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="learning_rate",
            type=float,
            description="Define a learning rate for the optimizer.",
            category=model_category,
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
            description="Define one or more dropout rates to be applied in the model (if applicable).",
            category=model_category,
            required=False,
            default=0.25,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(
                gte=0,
                lt=1
            )
        ),
        ConfigOption(
            name="epsilon",
            type=float,
            description="Define an epsilon value for the optimizer algorithm.",
            category=model_category,
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
            description="Choose an loss function from the available predefined losses.",
            category=model_category,
            required=False,
            default="mean_squared_error" if protocol in Protocol.regression_protocols() else "cross_entropy_loss",
            constraints=ConfigConstraints(
                allowed_values=list(get_available_losses_dict()[protocol].keys())
            )
        ),
        ConfigOption(
            name="disable_pytorch_compile",
            type=bool,
            description="Disable the automatic compilation of PyTorch models, "
                        "which can be useful for debugging or when encountering compatibility issues.",
            category=model_category,
            required=False,
            default=True
        )
    ]
