import os
import logging

from pathlib import Path

from biotrainer.datasets import TrainingDataLoader
from biotrainer.utilities import seed_all, read_config_file, count_parameters

logger = logging.getLogger(__name__)


def _validate_file(file_path: str):
    """
    Verify if a file exists and is not empty.
    Parameters
    ----------
    file_path : str
        Path to file to check
    Returns
    -------
    bool
        True if file exists and is non-zero size,
        False otherwise.
    """
    try:
        if os.stat(file_path).st_size == 0:
            raise Exception(f"The file at '{file_path}' is empty")
    except (OSError, TypeError) as e:
        raise Exception(f"The configuration file at '{file_path}' does not exist") from e


def parse_config_file_and_execute_run(config_file_path: str):
    _validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)

    execute(**config)


def execute(
        sequence_file: str, labels_file: str,
        model_choice: str = "CNN", num_epochs: int = 200,
        use_class_weights: bool = False, learning_rate: float = 1e-3,
        batch_size: int = 128, embedder_name: str = "prottrans_t5_xl_u50",
):
    seed_all()
    root = Path.cwd()

    training_data_loader = TrainingDataLoader(
        sequence_file,
        labels_file,
        embedder_name=embedder_name,
        embeddings_file_path=None,
        device=None
    )

    # Task-specific parameters
    n_classes = training_data_loader.get_number_of_classes()
    n_features = training_data_loader.get_number_of_features()

    if use_class_weights:
        class_weights = training_data_loader.get_class_weights()

    # create log directory if it does not exist yet
    log_root = root / 'log_CNN'
    if not log_root.is_dir():
        logger.info("Creating new log-directory: {}".format(log_root))
        log_root.mkdir()

    # experiment name will always hold information on model_name and architecture
    experiment_name = f"{embedder_name}_{model_choice}"
    logger.info(f'########### Experiment: {experiment_name} ###########')
    # Create directory for experiment logging results
    log_dir = log_root / experiment_name
    if not log_dir.is_dir():
        logger.info(f"Creating new log-directory: {log_dir}")
        log_dir.mkdir()

    train_loader = get_dataloader(train, batch_size=batch_size)
    test_loader = get_dataloader(test, batch_size=batch_size)
    val_loader = get_dataloader(val, batch_size=batch_size)

    early_stopper = EarlyStopper(log_dir)
    model = get_model(model_choice, n_classes, n_features)

    n_free_paras = count_parameters(model)
    logger.info(f'Experiment: {experiment_name}. Number of free parameters: {n_free_paras}')

    if use_class_weights:
        crit = nn.CrossEntropyLoss(weight=class_weights)
    else:
        crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    start = time.time()
    for epoch in range(num_epochs):  # for each epoch: train & test
        stop = early_stopper.check_performance(model, test_loader, crit, optimizer, epoch, num_epochs)
        if stop:  # if early stopping criterion was reached
            break
        training(model, train_loader, optimizer, crit, train_loader, epoch, num_epochs)

    # re-initialize the model to avoid any undesired information leake and only load checkpoint weights
    model = get_model(model_choice, n_classes, n_features)
    # load the model weights of the best checkpoint
    model = early_stopper.load_checkpoint(model)[0]

    end = time.time()
    print('Total training time: {:.1f}[m]'.format((end - start) / 60))
    print('Running final evaluation on the best checkpoint.')

    # XXMH: those are todos for myself not for you @Chris :)
    # TODO: write summary test set performance to file
    # TODO: add balanced accuracy etc (scikit compatibility in general)
    testing(model, val_loader, crit, epoch, epoch, log_dir=log_dir,
            set_name="val", get_confmat=True, class_labels=datasplitter.class_labels)
    testing(model, test_loader, crit, epoch, epoch, log_dir=log_dir,
            set_name="test", get_confmat=True, class_labels=datasplitter.class_labels)

    return None