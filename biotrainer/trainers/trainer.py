import time
import torch
import logging
import datetime

from pathlib import Path
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Union, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from ..losses import get_loss
from ..solvers import get_solver
from ..optimizers import get_optimizer
from ..datasets import get_collate_function
from ..models import get_model, count_parameters
from ..utilities import seed_all, get_device, SanityChecker

from .TargetManager import TargetManager
from .target_manager_utils import revert_mappings
from .embeddings import compute_embeddings, load_embeddings

logger = logging.getLogger(__name__)
output_vars = dict()


def training_and_evaluation_routine(
        # Needed
        sequence_file: str,
        # Defined previously
        protocol: str, output_dir: str, log_dir: str,
        # Optional with defaults
        labels_file: Optional[str] = None, mask_file: Optional[str] = None,
        model_choice: str = "CNN", num_epochs: int = 200,
        use_class_weights: bool = False, learning_rate: float = 1e-3,
        batch_size: int = 128, embedder_name: str = "custom_embeddings",
        embeddings_file: str = None,
        shuffle: bool = True, seed: int = 42, loss_choice: str = "cross_entropy_loss",
        optimizer_choice: str = "adam", patience: int = 10, epsilon: float = 0.001,
        device: Union[None, str, torch.device] = None,
        pretrained_model: str = None,
        save_test_predictions: bool = False,
        ignore_file_inconsistencies: bool = False,
        limited_sample_size: int = -1,
        # Everything else
        **kwargs
) -> Dict[str, Any]:
    global output_vars
    output_vars = deepcopy(locals())
    output_vars.pop('kwargs')

    # Initialization
    seed_all(seed)
    output_dir = Path(output_dir)

    # Get device
    device = get_device(device)
    output_vars['device'] = str(device)
    logger.info(f"Using device: {device}")

    # Generate embeddings if necessary, otherwise use existing embeddings and overwrite embedder_name
    if not embeddings_file or not Path(embeddings_file).is_file():
        embeddings_file = compute_embeddings(
            embedder_name=embedder_name, sequence_file=sequence_file,
            protocol=protocol, output_dir=output_dir
        )
        # Add to outconfig
        output_vars['embeddings_file'] = embeddings_file
    else:
        logger.info(f'Embeddings file was found at {embeddings_file}. Embeddings have not been computed.')
        embedder_name = f"precomputed_{Path(embeddings_file).stem}_{embedder_name}"

    # Mapping from id to embeddings
    id2emb = load_embeddings(embeddings_file_path=embeddings_file)

    # Find out feature size and add to output vars + logging
    embeddings_length = list(id2emb.values())[0].shape[-1]  # Last position in shape is always embedding length
    output_vars['n_features'] = embeddings_length
    logger.info(f"Number of features: {embeddings_length}")

    # Get datasets
    target_manager = TargetManager(protocol=protocol, sequence_file=sequence_file,
                                   labels_file=labels_file, mask_file=mask_file,
                                   ignore_file_inconsistencies=ignore_file_inconsistencies,
                                   limited_sample_size=limited_sample_size)
    train_dataset, val_dataset, test_dataset = target_manager.get_datasets(id2emb)
    output_vars['n_training_ids'] = len(target_manager.training_ids)
    output_vars['n_validation_ids'] = len(target_manager.validation_ids)
    output_vars['n_testing_ids'] = len(target_manager.testing_ids)
    output_vars['n_classes'] = target_manager.number_of_outputs

    # Get x_to_class specific logs and weights
    class_weights = None
    if 'class' in protocol or '_interaction' in protocol:
        output_vars['class_int_to_string'] = target_manager.class_int2str
        output_vars['class_str_to_int'] = target_manager.class_str2int
        logger.info(f"Number of classes: {output_vars['n_classes']}")
        # Compute class weights to pass as bias to model if option is set
        if use_class_weights:
            class_weights = target_manager.compute_class_weights()

    # Create the dataloaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        collate_fn=get_collate_function(protocol)
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        collate_fn=get_collate_function(protocol)
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        collate_fn=get_collate_function(protocol)
    )

    # Initialize model
    model = get_model(
        protocol=protocol, model_choice=model_choice,
        n_classes=output_vars['n_classes'], n_features=output_vars['n_features']
    )

    # Count and log number of free params
    n_free_parameters = count_parameters(model)
    output_vars['n_free_parameters'] = n_free_parameters

    # Initialize loss function
    loss_function = get_loss(
        protocol=protocol, loss_choice=loss_choice, device=device, weight=class_weights
    )

    # Initialize optimizer
    optimizer = get_optimizer(
        protocol=protocol, optimizer_choice=optimizer_choice,
        learning_rate=learning_rate, model_parameters=model.parameters()
    )

    # Tensorboard writer
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    writer.add_hparams({
        'model': model_choice,
        'num_epochs': num_epochs,
        'use_class_weights': use_class_weights,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'embedder_name': embedder_name,
        'seed': seed,
        'loss': loss_choice,
        'optimizer': optimizer_choice,
    }, {})

    # Create solver
    solver = get_solver(
        protocol, network=model, optimizer=optimizer, loss_function=loss_function, device=device,
        number_of_epochs=num_epochs, patience=patience, epsilon=epsilon, log_writer=writer, experiment_dir=log_dir,
        num_classes=output_vars['n_classes']
    )

    if pretrained_model:
        solver.load_checkpoint(checkpoint_path=pretrained_model)

    # Perform training of the model (time intensive -- where things can go wrong)
    _do_and_log_training(solver, train_loader, val_loader)

    # Finally, run evaluation of test set
    _do_and_log_evaluation(solver, test_loader, target_manager, save_test_predictions)

    # Do sanity check TODO: Think about purpose and pros and cons, flags in config, tests..
    sanity_checker = SanityChecker(output_vars=output_vars, mode="Warn")
    sanity_checker.check_test_results()

    return output_vars


def _do_and_log_training(solver, train_loader, val_loader):
    start_time_abs = datetime.datetime.now()
    start_time = time.perf_counter()
    epoch_iterations = solver.train(train_loader, val_loader)
    end_time = time.perf_counter()
    end_time_abs = datetime.datetime.now()

    # Logging
    logger.info(f'Total training time: {(end_time - start_time) / 60:.1f}[m]')

    # Save training time for prosperity
    output_vars['start_time'] = start_time_abs
    output_vars['end_time'] = end_time_abs
    output_vars['elapsed_time'] = end_time - start_time

    # Save metrics from best training epoch
    output_vars['training_iteration_result_best_epoch'] = epoch_iterations[solver.get_best_epoch()]


def _do_and_log_evaluation(solver, test_loader, target_manager, save_test_predictions):
    # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
    logger.info('Running final evaluation on the best checkpoint.')

    solver.load_checkpoint()
    test_results = solver.inference(test_loader, calculate_test_metrics=True)

    if save_test_predictions:
        test_results['predictions'] = revert_mappings(protocol=target_manager.protocol,
                                                      test_predictions=test_results['predictions'],
                                                      class_int2str=target_manager.class_int2str)
        output_vars['test_iterations_results'] = test_results
    else:
        output_vars['test_iterations_results'] = test_results['metrics']

    logger.info(f"Test set metrics: {test_results['metrics']}")
    logger.info(f"Extensive output information can be found at {output_vars['output_dir']}/out.yml")
