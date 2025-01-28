from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol

def training_config(protocol: Protocol):
    return "", [
        ConfigOption(
            name="num_epochs",
            type=int,
            required=False,
            default=200,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(gt=0)
        ),
        ConfigOption(
            name="batch_size",
            type=int,
            required=False,
            default=128,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(gt=0)
        ),
        ConfigOption(
            name="patience",
            type=int,
            required=False,
            default=10,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(gt=0)
        ),
        ConfigOption(
            name="shuffle",
            type=bool,
            required=False,
            default=True
        ),
        ConfigOption(
            name="use_class_weights",
            type=bool,
            required=False,
            default=False,
            allow_hyperparameter_optimization=True,
        ),
        ConfigOption(
            name="auto_resume",
            type=bool,
            required=False,
            default=False
        ),
        ConfigOption(
            name="pretrained_model",
            type=str,
            required=False,
            default="",
            constraints=ConfigConstraints(allowed_formats=["safetensors"])
        ),
        ConfigOption(
            name="limited_sample_size",
            type=int,
            required=False,
            default=-1,
            constraints=ConfigConstraints(
                gte=-1  # Allows -1 or positive values
            )
        )
    ]