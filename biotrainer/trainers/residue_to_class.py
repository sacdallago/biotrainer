import time
import logging

from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader

from ..solvers import ResidueSolver
from ..datasets import TrainingDatasetLoader, pad_sequences
from ..utilities import seed_all, count_parameters
from ..models import get_model
from ..losses import get_loss
from ..optimizers import get_optimizer

logger = logging.getLogger(__name__)


def residue_to_class(
        # Needed
        sequence_file: str, labels_file: str,
        # Defined previously
        protocol: str, output_dir: str,
        # Optional with defaults
        model_choice: str = "CNN", num_epochs: int = 200,
        use_class_weights: bool = False, learning_rate: float = 1e-3,
        batch_size: int = 128, embedder_name: str = "prottrans_t5_xl_u50",
        shuffle: bool = True, seed: int = 42, loss_choice: str = "cross_entropy_loss",
        optimizer_choice: str = "adam", patience: int = 10, epsilon: float = 0.001,
        # Everything else
        **kwargs
):
    output_vars = deepcopy(locals())
    output_vars.pop('kwargs')

    seed_all(seed)
    output_dir = Path(output_dir)

    experiment_name = f"{embedder_name}_{model_choice}"

    # create log directory if it does not exist yet
    logger.info(f'########### Experiment: {experiment_name} ###########')
    log_dir = output_dir / model_choice / embedder_name

    if not log_dir.is_dir():
        logger.info(f"Creating log-directory: {log_dir}")
        log_dir.mkdir(parents=True)
        output_vars['log_dir'] = log_dir.name

    training_data_loader = TrainingDatasetLoader(
        sequence_file,
        labels_file,
        embedder_name=embedder_name,
        embeddings_file_path=None,
        device=None
    )

    # Task-specific parameters
    n_classes = training_data_loader.get_number_of_classes()
    output_vars['n_classes'] = n_classes
    n_features = training_data_loader.get_number_of_features()
    output_vars['n_features'] = n_features

    if use_class_weights:
        class_weights = training_data_loader.get_class_weights()
    else:
        class_weights = None

    train_loader = DataLoader(
        dataset=training_data_loader.get_training_dataset(), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=pad_sequences
    )
    val_loader = DataLoader(
        dataset=training_data_loader.get_validation_dataset(), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=pad_sequences
    )
    test_loader = DataLoader(
        dataset=training_data_loader.get_testing_dataset(), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=pad_sequences
    )

    model = get_model(protocol=protocol, model_choice=model_choice, n_classes=n_classes, n_features=n_features)
    loss_function = get_loss(protocol=protocol, loss_choice=loss_choice, weight=class_weights)
    optimizer = get_optimizer(protocol=protocol, optimizer_choice=optimizer_choice, learning_rate=learning_rate,
                              model_parameters=model.parameters())

    solver = ResidueSolver(network=model, optimizer=optimizer, loss_function=loss_function,
                           number_of_epochs=num_epochs, patience=patience, epsilon=epsilon)

    n_free_parameters = count_parameters(model)

    logger.info(f'Experiment: {experiment_name}. Number of free parameters: {n_free_parameters}')
    output_vars['n_free_parameters'] = n_free_parameters

    start = time.time()
    epoch_iteration_results = solver.train(train_loader, val_loader)
    output_vars['training_iteration_results'] = epoch_iteration_results

    # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
    solver.load_checkpoint()

    test_results = solver.inference(test_loader)
    output_vars['test_iterations_results'] = test_results

    end = time.time()
    logger.info(f'Total training time: {(end - start) / 60:.1f}[m]')
    logger.info('Running final evaluation on the best checkpoint.')
    output_vars['start_time'] = start
    output_vars['end_time'] = end
    output_vars['elapsed_time'] = end-start

    return output_vars
