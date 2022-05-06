import h5py
import time
import torch
import logging
import itertools

from copy import deepcopy
from pathlib import Path
from typing import Dict, Union
from collections import Counter
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from bio_embeddings.utilities.pipeline import execute_pipeline_from_config

from ..solvers import get_solver
from ..datasets import get_dataset
from ..utilities import seed_all, get_device
from ..utilities.config import write_config_file
from ..models import get_model, count_parameters
from ..losses import get_loss
from ..optimizers import get_optimizer

logger = logging.getLogger(__name__)


class Trainer(ABC):

    @staticmethod
    @abstractmethod
    def pipeline(**kwargs):
        pass

    def _execute_pipeline(self,
                          # Needed
                          sequence_file: str,
                          # Defined previously
                          protocol: str, output_dir: str,
                          # Optional with defaults
                          labels_file: str = "",
                          model_choice: str = "CNN", num_epochs: int = 200,
                          use_class_weights: bool = False, learning_rate: float = 1e-3,
                          batch_size: int = 128, embedder_name: str = "prottrans_t5_xl_u50",
                          embeddings_file_path: str = None,
                          shuffle: bool = True, seed: int = 42, loss_choice: str = "cross_entropy_loss",
                          optimizer_choice: str = "adam", patience: int = 10, epsilon: float = 0.001,
                          device: Union[None, str, torch.device] = None,
                          # Everything else
                          **kwargs
                          ):
        output_dir, output_vars, experiment_name, log_dir = self._setup(seed, output_dir, embeddings_file_path,
                                                                        embedder_name, model_choice, device)

        training_ids, validation_ids, testing_ids, id2label, id2fasta = self._load_sequences_and_labels(sequence_file,
                                                                                                        labels_file)

        if len(training_ids) < 1 or len(validation_ids) < 1 or len(testing_ids) < 1:
            raise ValueError("Not enough samples for training, validation and testing!")

        output_vars['training_ids'] = training_ids
        output_vars['validation_ids'] = validation_ids
        output_vars['testing_ids'] = testing_ids

        class_labels, id2label, class_int2str, class_str2int = self._generate_class_labels(id2label, id2fasta)

        output_vars['class_int_to_string'] = class_int2str
        output_vars['class_string_to_integer'] = class_str2int

        id2emb = self._load_embeddings(embeddings_file_path, sequence_file, embedder_name, output_dir)

        output_vars['n_classes'] = len(class_labels)
        logger.info(f"Number of classes: {output_vars['n_classes']}")

        output_vars['n_features'] = self._get_number_features(id2emb, training_ids)
        logger.info(f"Number of features: {output_vars['n_features']}")

        if use_class_weights:
            class_weights = self._get_class_weights(id2label, class_str2int, class_int2str)
        else:
            class_weights = None

        train_loader, val_loader, test_loader = self._get_dataloaders(protocol, id2emb, id2label,
                                                                      training_ids, validation_ids, testing_ids,
                                                                      batch_size, shuffle)

        model, loss_function, optimizer = self._get_model_and_training_params(protocol, model_choice, output_vars,
                                                                              loss_choice, class_weights,
                                                                              optimizer_choice, learning_rate)

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
        solver = get_solver(protocol,
                            network=model, optimizer=optimizer, loss_function=loss_function, device=device,
                            number_of_epochs=num_epochs, patience=patience, epsilon=epsilon, log_writer=writer
                            )

        # Count and log number of free params
        n_free_parameters = count_parameters(model)
        logger.info(f'Experiment: {experiment_name}. Number of free parameters: {n_free_parameters}')
        output_vars['n_free_parameters'] = n_free_parameters

        # Training
        start_time, end_time = self._do_training(solver, train_loader, val_loader)

        logger.info(f'Total training time: {(end_time - start_time) / 60:.1f}[m]')
        output_vars['start_time'] = start_time
        output_vars['end_time'] = end_time
        output_vars['elapsed_time'] = end_time - start_time

        # Evaluation
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        solver.load_checkpoint()

        logger.info('Running final evaluation on the best checkpoint.')
        test_results = solver.inference(test_loader)
        output_vars['test_iterations_results'] = test_results

        write_config_file(
            str(log_dir / "out.yml"),
            output_vars
        )

        return output_vars

    @staticmethod
    def _setup(seed, output_dir, embeddings_file_path, embedder_name, model_choice, device):
        output_vars = deepcopy(locals())

        seed_all(seed)
        output_dir = Path(output_dir)

        if embeddings_file_path:
            embedder_name = f"precomputed_{Path(embeddings_file_path).stem}_{embedder_name}"

        experiment_name = f"{embedder_name}_{model_choice}"

        # create log directory if it does not exist yet
        logger.info(f'########### Experiment: {experiment_name} ###########')
        log_dir = output_dir / model_choice / embedder_name

        if not log_dir.is_dir():
            logger.info(f"Creating log-directory: {log_dir}")
            log_dir.mkdir(parents=True)
            output_vars['log_dir'] = str(log_dir)

        # Get device
        device = get_device(device)
        output_vars['device'] = str(device)
        return output_dir, output_vars, experiment_name, log_dir

    @abstractmethod
    def _load_sequences_and_labels(self, sequence_file, labels_file):
        pass

    @abstractmethod
    def _generate_class_labels(self, id2label, id2fasta):
        pass

    @abstractmethod
    def _get_embeddings_config_and_file_name(self, sequence_file, output_dir, embedder_name):
        pass

    @staticmethod
    def _get_class_weights(id2label: Dict[str, str], class_str2int: Dict[str, int],
                           class_int2str: Dict[int, str]) -> torch.FloatTensor:
        # concatenate all labels irrespective of protein to count class sizes
        counter = Counter(list(itertools.chain.from_iterable(
            [list(labels) for labels in id2label.values()]
        )))
        # total number of samples in the set irrespective of classes
        n_samples = sum([counter[idx] for idx in range(len(class_str2int))])
        # balanced class weighting (inversely proportional to class size)
        class_weights = [
            (n_samples / (len(class_str2int) * counter[idx])) for idx in range(len(class_str2int))
        ]

        logger.info(f"Total number of samples/residues: {n_samples}")
        logger.info("Individual class counts and weights:")
        for c in counter:
            logger.info(f"\t{class_int2str[c]} : {counter[c]} ({class_weights[c]:.3f})")

        return torch.FloatTensor(class_weights)

    def _load_embeddings(self, embeddings_file_path, sequence_file, embedder_name, output_dir):
        # If embeddings don't exist, create them using the bio_embeddings pipeline
        if not embeddings_file_path or not Path(embeddings_file_path).is_file():
            embeddings_config, embeddings_file_name = self._get_embeddings_config_and_file_name(sequence_file,
                                                                                                output_dir,
                                                                                                embedder_name)
            # Check if bio-embeddings has already been run
            embeddings_file_path = str(
                Path(embeddings_config['global']['prefix']) / "embeddings" / embeddings_file_name)

            if not Path(embeddings_file_path).is_file():
                _ = execute_pipeline_from_config(embeddings_config, overwrite=False)

        # load pre-computed embeddings in .h5 file format computed via bio_embeddings
        logger.info(f"Loading embeddings from: {embeddings_file_path}")
        start = time.time()

        # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
        embeddings_file = h5py.File(embeddings_file_path, 'r', rdcc_nbytes=1024 ** 2 * 4000, rdcc_nslots=1e7)
        id2emb = {embeddings_file[idx].attrs["original_id"]: embedding for (idx, embedding) in embeddings_file.items()}

        # Logging
        logger.info(f"Read {len(id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.time() - start):.1f}[s]")

        return id2emb

    @abstractmethod
    def _get_number_features(self, id2emb, training_ids):
        pass

    @abstractmethod
    def _get_collate_function(self):
        pass

    def _get_dataloaders(self, protocol, id2emb, id2label,
                         training_ids, validation_ids, testing_ids,
                         batch_size, shuffle):
        train_dataset = get_dataset(protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(id2label[idx])) for idx in training_ids
        })
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
            collate_fn=self._get_collate_function()
        )

        # Validation
        val_dataset = get_dataset(protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(id2label[idx])) for idx in validation_ids
        })
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
            collate_fn=self._get_collate_function()
        )

        # Test
        test_dataset = get_dataset(protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(id2label[idx])) for idx in testing_ids
        })
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
            collate_fn=self._get_collate_function()
        )
        return train_loader, val_loader, test_loader

    def _get_model_and_training_params(self, protocol, model_choice, output_vars,
                                       loss_choice, class_weights,
                                       optimizer_choice, learning_rate):
        model = get_model(
            protocol=protocol, model_choice=model_choice,
            n_classes=output_vars['n_classes'], n_features=output_vars['n_features']
        )
        loss_function = get_loss(
            protocol=protocol, loss_choice=loss_choice, weight=class_weights
        )
        optimizer = get_optimizer(
            protocol=protocol, optimizer_choice=optimizer_choice,
            learning_rate=learning_rate, model_parameters=model.parameters()
        )

        return model, loss_function, optimizer

    @staticmethod
    def _do_training(solver, train_loader, val_loader):
        start_time = time.time()
        _ = solver.train(train_loader, val_loader)
        end_time = time.time()
        return start_time, end_time
