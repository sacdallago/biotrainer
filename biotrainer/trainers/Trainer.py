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

    def __init__(self,
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
        self.sequence_file = sequence_file
        self.protocol = protocol
        self.output_dir = Path(output_dir)
        self.labels_file = labels_file
        self.model_choice = model_choice
        self.num_epochs = num_epochs
        self.use_class_weights = use_class_weights
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.embedder_name = embedder_name
        self.embeddings_file_path = embeddings_file_path
        self.shuffle = shuffle
        self.seed = seed
        self.loss_choice = loss_choice
        self.optimizer_choice = optimizer_choice
        self.patience = patience
        self.epsilon = epsilon
        self.device = device
        self.output_vars: Dict[str,] = dict()
        self.log_dir = None
        self.experiment_name = ""
        self.training_ids = list()
        self.validation_ids = list()
        self.class_labels = set()
        self.id2label: Dict[str, str] = deepcopy(locals())
        self.id2fasta = dict()
        self.id2emb = dict()
        self.class_str2int: Dict[str, int] = dict()
        self.class_int2str: Dict[int, str] = dict()
        self.testing_ids = list()
        self.class_weights = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.writer = None
        self.solver = None

    @staticmethod
    @abstractmethod
    def pipeline(**kwargs):
        pass

    def _execute_pipeline(self):
        self._setup()

        self._load_sequences_and_labels()

        if len(self.training_ids) < 1 or len(self.validation_ids) < 1 or len(self.testing_ids) < 1:
            raise ValueError("Not enough samples for training, validation and testing!")

        self.output_vars['training_ids'] = self.training_ids
        self.output_vars['validation_ids'] = self.validation_ids
        self.output_vars['testing_ids'] = self.testing_ids

        self._generate_class_labels()

        self.output_vars['class_int_to_string'] = self.class_int2str
        self.output_vars['class_string_to_integer'] = self.class_str2int
        self.output_vars['n_classes'] = len(self.class_labels)
        logger.info(f"Number of classes: {self.output_vars['n_classes']}")

        # Load embeddings
        self._load_embeddings()

        # Get number of input features
        self.output_vars['n_features'] = self._get_number_features()
        logger.info(f"Number of features: {self.output_vars['n_features']}")

        if self.use_class_weights:
            self._compute_class_weights()

        # Create dataloaders and model
        self._create_dataloaders()
        self._create_model_and_training_params()

        # Tensorboard writer
        self._create_writer()

        # Create solver
        self.solver = get_solver(self.protocol,
                                 network=self.model, optimizer=self.optimizer, loss_function=self.loss_function,
                                 device=self.device, number_of_epochs=self.num_epochs,
                                 patience=self.patience, epsilon=self.epsilon, log_writer=self.writer
                                 )

        # Count and log number of free params
        self._log_number_free_params()

        # Do training and log time
        self._do_and_log_training()

        # Do evaluation and log results
        self._do_and_log_evaluation()

        # Save configuration data
        write_config_file(str(self.log_dir / "out.yml"), self.output_vars)

        return self.output_vars

    def _setup(self):
        seed_all(self.seed)

        if self.embeddings_file_path:
            self.embedder_name = f"precomputed_{Path(self.embeddings_file_path).stem}_{self.embedder_name}"

        self.experiment_name = f"{self.embedder_name}_{self.model_choice}"
        logger.info(f'########### Experiment: {self.experiment_name} ###########')

        # create log directory if it does not exist yet
        self.log_dir = self.output_dir / self.model_choice / self.embedder_name
        if not self.log_dir.is_dir():
            logger.info(f"Creating log-directory: {self.log_dir}")
            self.log_dir.mkdir(parents=True)
            self.output_vars['log_dir'] = str(self.log_dir)

        # Get device
        self.device = get_device(self.device)
        self.output_vars['device'] = str(self.device)
        self.output_vars['output_dir'] = self.output_dir

    @abstractmethod
    def _load_sequences_and_labels(self):
        """
        This method must load the sequences and labels from the provided sequence (and label) file(s).
        It must set the following member attributes:
        self.id2fasta
        self.id2label
        self.training_ids
        self.validation_ids
        self.testing_ids
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_class_labels(self):
        """
        This method must generate the class labels.
        It must set the following member attributes:
        self.class_labels
        self.class_str2int
        self.class_int2str
        self.id2label
        """
        raise NotImplementedError

    @abstractmethod
    def _use_reduced_embeddings(self) -> bool:
        """
        Define if reduced embeddings from bio_embeddings should be used.
        Reduced means that the per-residue embeddings are reduced to a per-sequence embedding
        Returns
        -------
        True: Use reduced embeddings from bio_embeddings
        False: Use non-reduced embeddings
        """
        raise NotImplementedError

    def _compute_class_weights(self):
        # concatenate all labels irrespective of protein to count class sizes
        counter = Counter(list(itertools.chain.from_iterable(
            [list(labels) for labels in self.id2label.values()]
        )))
        # total number of samples in the set irrespective of classes
        n_samples = sum([counter[idx] for idx in range(len(self.class_str2int))])
        # balanced class weighting (inversely proportional to class size)
        class_weights = [
            (n_samples / (len(self.class_str2int) * counter[idx])) for idx in range(len(self.class_str2int))
        ]

        logger.info(f"Total number of samples/residues: {n_samples}")
        logger.info("Individual class counts and weights:")
        for c in counter:
            logger.info(f"\t{self.class_int2str[c]} : {counter[c]} ({class_weights[c]:.3f})")
        self.class_weights = torch.FloatTensor(class_weights)

    def _load_embeddings(self):
        # If embeddings don't exist, create them using the bio_embeddings pipeline
        if not self.embeddings_file_path or not Path(self.embeddings_file_path).is_file():
            use_reduced_embeddings = self._use_reduced_embeddings()
            embeddings_config = {
                "global": {
                    "sequences_file": self.sequence_file,
                    "prefix": str(self.output_dir / self.embedder_name),
                    "simple_remapping": True
                },
                "embeddings": {
                    "type": "embed",
                    "protocol": self.embedder_name,
                    "reduce": use_reduced_embeddings,
                    "discard_per_amino_acid_embeddings": use_reduced_embeddings
                }
            }
            embeddings_file_name = "reduced_embeddings_file.h5" if use_reduced_embeddings else "embeddings_file.h5"
            # Check if bio-embeddings has already been run
            self.embeddings_file_path = str(
                Path(embeddings_config['global']['prefix']) / "embeddings" / embeddings_file_name)

            if not Path(self.embeddings_file_path).is_file():
                _ = execute_pipeline_from_config(embeddings_config, overwrite=False)

        # load pre-computed embeddings in .h5 file format computed via bio_embeddings
        logger.info(f"Loading embeddings from: {self.embeddings_file_path}")
        start = time.time()

        # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
        embeddings_file = h5py.File(self.embeddings_file_path, 'r', rdcc_nbytes=1024 ** 2 * 4000, rdcc_nslots=1e7)
        self.id2emb = {embeddings_file[idx].attrs["original_id"]: embedding for (idx, embedding) in
                       embeddings_file.items()}

        # Logging
        logger.info(f"Read {len(self.id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.time() - start):.1f}[s]")

    @abstractmethod
    def _get_number_features(self) -> int:
        """
        Returns
        -------
        Number of input features for the model (=> shape of the embedding vector).
        """
        pass

    @abstractmethod
    def _get_collate_function(self):
        """
        If the dataloaders use a collate_function, it must be returned by this method.
        """
        pass

    def _create_writer(self):
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))
        self.writer.add_hparams({
            'model': self.model_choice,
            'num_epochs': self.num_epochs,
            'use_class_weights': self.use_class_weights,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'embedder_name': self.embedder_name,
            'seed': self.seed,
            'loss': self.loss_choice,
            'optimizer': self.optimizer_choice,
        }, {})

    def _log_number_free_params(self):
        n_free_parameters = count_parameters(self.model)
        logger.info(f'Experiment: {self.experiment_name}. Number of free parameters: {n_free_parameters}')
        self.output_vars['n_free_parameters'] = n_free_parameters

    def _create_dataloaders(self):
        train_dataset = get_dataset(self.protocol, {
            idx: (torch.tensor(self.id2emb[idx]), torch.tensor(self.id2label[idx])) for idx in self.training_ids
        })
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False,
            collate_fn=self._get_collate_function()
        )

        # Validation
        val_dataset = get_dataset(self.protocol, {
            idx: (torch.tensor(self.id2emb[idx]), torch.tensor(self.id2label[idx])) for idx in self.validation_ids
        })
        self.val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False,
            collate_fn=self._get_collate_function()
        )

        # Test
        test_dataset = get_dataset(self.protocol, {
            idx: (torch.tensor(self.id2emb[idx]), torch.tensor(self.id2label[idx])) for idx in self.testing_ids
        })
        self.test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False,
            collate_fn=self._get_collate_function()
        )

    def _create_model_and_training_params(self):
        self.model = get_model(
            protocol=self.protocol, model_choice=self.model_choice,
            n_classes=self.output_vars['n_classes'], n_features=self.output_vars['n_features']
        )
        self.loss_function = get_loss(
            protocol=self.protocol, loss_choice=self.loss_choice, weight=self.class_weights
        )
        self.optimizer = get_optimizer(
            protocol=self.protocol, optimizer_choice=self.optimizer_choice,
            learning_rate=self.learning_rate, model_parameters=self.model.parameters()
        )

    def _do_and_log_training(self):
        start_time = time.time()
        _ = self.solver.train(self.train_loader, self.val_loader)
        end_time = time.time()
        logger.info(f'Total training time: {(end_time - start_time) / 60:.1f}[m]')
        self.output_vars['start_time'] = start_time
        self.output_vars['end_time'] = end_time
        self.output_vars['elapsed_time'] = end_time - start_time

    def _do_and_log_evaluation(self):
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        self.solver.load_checkpoint()

        logger.info('Running final evaluation on the best checkpoint.')
        test_results = self.solver.inference(self.test_loader)
        self.output_vars['test_iterations_results'] = test_results
