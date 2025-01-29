from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol

def training_config(protocol: Protocol):
    training_category = "training"
    return "", [
        ConfigOption(
            name="num_epochs",
            type=int,
            description="Define the number of epochs to train the model.",
            category=training_category,
            required=False,
            default=200,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(gt=0)
        ),
        ConfigOption(
            name="batch_size",
            type=int,
            description="Define the batch size used for model training.",
            category=training_category,
            required=False,
            default=128,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(gt=0)
        ),
        ConfigOption(
            name="patience",
            type=int,
            description="Define the number of epochs to wait before early stopping.",
            category=training_category,
            required=False,
            default=10,
            allow_hyperparameter_optimization=True,
            constraints=ConfigConstraints(gt=0)
        ),
        ConfigOption(
            name="shuffle",
            type=bool,
            description="Define whether to shuffle the training data before each training epoch.",
            category=training_category,
            required=False,
            default=True
        ),
        ConfigOption(
            name="use_class_weights",
            type=bool,
            description="Define whether to use class weights for training, "
                        "which can be useful for handling imbalanced datasets.",
            category=training_category,
            required=False,
            default=False,
            allow_hyperparameter_optimization=True,
        ),
        ConfigOption(
            name="auto_resume",
            type=bool,
            description="Define whether to to automatically resume training from the "
                        "last checkpoint in case of interruptions.",
            category=training_category,
            required=False,
            default=False
        ),
        ConfigOption(
            name="pretrained_model",
            type=str,
            description="Define if a pretrained model should be loaded.",
            category=training_category,
            required=False,
            default="",
            constraints=ConfigConstraints(allowed_formats=["safetensors"])
        ),
        ConfigOption(
            name="limited_sample_size",
            type=int,
            description="Define a maximum number of samples to be used for training from the training dataset. "
                        "Useful for example to test, if a model is able to overfit. "
                        "A value of -1 indicates no limit.",
            category=training_category,
            required=False,
            default=-1,
            constraints=ConfigConstraints(
                gte=-1  # Allows -1 or positive values
            )
        )
    ]