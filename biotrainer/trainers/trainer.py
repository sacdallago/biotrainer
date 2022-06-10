import time
import torch
import logging

from copy import deepcopy
from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..datasets import get_dataset, get_collate_function
from ..utilities import seed_all, get_device, read_FASTA
from ..models import get_model, count_parameters
from ..losses import get_loss
from ..optimizers import get_optimizer
from ..solvers import get_solver

from .TargetManager import TargetManager
from .embeddings import compute_embeddings, load_embeddings
from .utilities import get_split_lists, get_class_weights


logger = logging.getLogger(__name__)


def training_and_evaluation_routine(
        # Needed
        sequence_file: str, labels_file: str,
        # Defined previously
        protocol: str, output_dir: str,
        # Optional with defaults
        model_choice: str = "CNN", num_epochs: int = 200,
        use_class_weights: bool = False, learning_rate: float = 1e-3,
        batch_size: int = 128, embedder_name: str = "prottrans_t5_xl_u50",
        embeddings_file: str = None,
        shuffle: bool = True, seed: int = 42, loss_choice: str = "cross_entropy_loss",
        optimizer_choice: str = "adam", patience: int = 10, epsilon: float = 0.001,
        device: Union[None, str, torch.device] = None,
        # Everything else
        **kwargs
):
    output_vars = deepcopy(locals())
    output_vars.pop('kwargs')

    # Initialization
    seed_all(seed)
    output_dir = Path(output_dir)

    # Create log directory if it does not exist yet
    log_dir = output_dir / model_choice / embedder_name
    if not log_dir.is_dir():
        logger.info(f"Creating log-directory: {log_dir}")
        log_dir.mkdir(parents=True)
    output_vars['log_dir'] = str(log_dir)

    # Get device
    device = get_device(device)
    output_vars['device'] = str(device)

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

    # Mapping from id to protein sequences
    protein_sequences = read_FASTA(sequence_file)
    id2sequence = {protein.id: str(protein.seq) for protein in protein_sequences}

    # Load targets
    target_manager = TargetManager(protocol=protocol, protein_sequences=protein_sequences, labels_file=labels_file)

    # Mapping from id to target(s)
    id2target = target_manager.id2target

    # This will be 1 for regression tasks, 2 for binary classification tasks, and N>2 for everything else
    output_vars['n_classes'] = target_manager.number_of_outputs

    # Only relevant for class prediction problems:
    #  1. Write the mappings from int to string and reverse into the outconfig
    #  2. Compute class weights to pass as bias to model if option is set
    class_weights = None

    if 'class' in protocol:
        output_vars['class_int_to_string'] = target_manager.class_int2str
        output_vars['class_string_to_integer'] = target_manager.class_str2int

        logger.info(f"Number of classes: {output_vars['n_classes']}")

        # Get loss weights
        if use_class_weights:
            class_weights = get_class_weights(
                id2target, class_str2int=target_manager.class_str2int, class_int2str=target_manager.class_int2str
            )

    # Get the data splits
    training_ids, validation_ids, testing_ids = get_split_lists(target_manager.id2attributes)

    # Write to outconfig for prosperity
    output_vars['training_ids'] = training_ids
    output_vars['validation_ids'] = validation_ids
    output_vars['testing_ids'] = testing_ids

    # Sanity check (at least one sample for each set!)
    if len(training_ids) < 1 or len(validation_ids) < 1 or len(testing_ids) < 1:
        raise ValueError("Not enough samples for training, validation and testing!")

    # Create the datasets:
    train_dataset = get_dataset(protocol, {
        idx: (torch.tensor(id2emb[idx]), torch.tensor(id2target[idx])) for idx in training_ids
    })
    val_dataset = get_dataset(protocol, {
        idx: (torch.tensor(id2emb[idx]), torch.tensor(id2target[idx])) for idx in validation_ids
    })
    test_dataset = get_dataset(protocol, {
        idx: (torch.tensor(id2emb[idx]), torch.tensor(id2target[idx])) for idx in testing_ids
    })

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
        protocol=protocol, loss_choice=loss_choice, weight=class_weights
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
        number_of_epochs=num_epochs, patience=patience, epsilon=epsilon, log_writer=writer, experiment_dir=log_dir
    )

    # Perform training of the model (time intensive -- where things can go wrong)
    start_time = time.time()
    _ = solver.train(train_loader, val_loader)
    end_time = time.time()

    # Logging
    logger.info(f'Total training time: {(end_time - start_time) / 60:.1f}[m]')

    # Save training time for prosperity
    output_vars['start_time'] = start_time
    output_vars['end_time'] = end_time
    output_vars['elapsed_time'] = end_time - start_time

    # Finally, run evaluation of test set

    # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
    logger.info('Running final evaluation on the best checkpoint.')

    solver.load_checkpoint()
    test_results = solver.inference(test_loader)

    # If residue-to-class problem, map the integers back to the class labels (single letters)
    if protocol == 'residue_to_class':
        test_results['predictions'] = ["".join(
            [target_manager.class_int2str[p] for p in prediction]
        ) for prediction in test_results['predictions']]

    # If sequence-to-class problem, map the integers back to the class labels (whatever length)
    elif protocol == "sequence_to_class":
        test_results['predictions'] = [target_manager.class_int2str[p] for p in test_results['predictions']]

    output_vars['test_iterations_results'] = test_results
