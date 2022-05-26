import torch
import logging
import itertools
import numpy as np
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset

from collections import Counter
from ..datasets import get_dataset
from ..utilities import read_FASTA, get_split_lists, get_attributes_from_seqrecords

logger = logging.getLogger(__name__)


class PredictionIOHandler:
    def __init__(self,
                 # Necessary
                 protocol: str, sequence_file: str,
                 # Optional:
                 labels_file: str = "",
                 **kwargs):
        self._protocol = protocol
        self._sequence_file = sequence_file
        self._labels_file = labels_file
        self._id2target: Dict[str, Any] = dict()
        self._id2attributes: Dict[str, Any] = dict()
        self._class_str2int: Dict[str, int] = dict()
        self._class_int2str: Dict[int, str] = dict()
        self._class_labels = set()

    def _handle_class_labels_on_residue_level(self):
        # Infer classes from data
        for classes in self._id2target.values():
            self._class_labels = self._class_labels | set(classes)

        # Create a mapping from integers to class labels and reverse
        self._class_str2int = {letter: idx for idx, letter in enumerate(self._class_labels)}
        self._class_int2str = {idx: letter for idx, letter in enumerate(self._class_labels)}

        # Convert label values to lists of numbers based on the maps
        self._id2target = {identifier: np.array([self._class_str2int[label] for label in labels])
                           for identifier, labels in self._id2target.items()}  # classes idxs (zero-based)

    def _handle_class_labels_on_sequence_level(self):
        # Infer classes from data
        self._class_labels = set(self._id2target.values())
        # Create a mapping from integers to class labels and reverse
        self._class_str2int = {letter: idx for idx, letter in enumerate(self._class_labels)}
        self._class_int2str = {idx: letter for idx, letter in enumerate(self._class_labels)}
        # Convert label values to lists of numbers based on the maps
        self._id2target = {identifier: np.array(self._class_str2int[label])
                           for identifier, label in self._id2target.items()}  # classes idxs (zero-based)

    def _log_class_labels(self, output_vars):
        output_vars['class_int_to_string'] = self._class_int2str
        output_vars['class_string_to_integer'] = self._class_str2int
        output_vars['n_classes'] = len(self._class_labels)
        logger.info(f"Number of classes: {output_vars['n_classes']}")

    def _validate_id2target(self, id2fasta):
        if 'residue_' in self._protocol:
            # Make sure the length of the sequences in the FASTA matches the length of the sequences in the labels
            for seq_id, seq in id2fasta.items():
                if len(seq) != self._id2target[seq_id].size:
                    Exception(f"Length mismatch for {seq_id}: Seq={len(seq)} VS Labels={self._id2target[seq_id].size}")

    def _calculate_id2target(self, protein_sequences, output_vars):
        # 1. Residue Level
        if 'residue_' in self._protocol:
            label_sequences = read_FASTA(self._labels_file)
            self._id2attributes = get_attributes_from_seqrecords(label_sequences)
            self._id2target = {label.id: str(label.seq) for label in label_sequences}
            # a) Class output
            if 'class' in self._protocol:
                # Infer classes from data
                self._handle_class_labels_on_residue_level()
                self._log_class_labels(output_vars)
            # b) Value output
            else:
                raise NotImplementedError
        # 2. Sequence Level
        else:
            self._id2attributes = get_attributes_from_seqrecords(protein_sequences)
            self._id2target = {seq_id: seq_vals["TARGET"] for seq_id, seq_vals in self._id2attributes.items()}
            # a) Class output
            if 'class' in self._protocol:
                # Infer classes from data
                self._handle_class_labels_on_sequence_level()
                self._log_class_labels(output_vars)
            # b) Value output
            else:
                raise NotImplementedError

        if not self._id2target:
            raise Exception("Prediction targets not found or could not be calculated!")

    def compute_loss_weights(self) -> torch.FloatTensor:
        if 'class' in self._protocol:
            # concatenate all labels irrespective of protein to count class sizes
            counter = Counter(list(itertools.chain.from_iterable(
                [list(labels) for labels in self._id2target.values()]
            )))
            # total number of samples in the set irrespective of classes
            n_samples = sum([counter[idx] for idx in range(len(self._class_str2int))])
            # balanced class weighting (inversely proportional to class size)
            class_weights = [
                (n_samples / (len(self._class_str2int) * counter[idx])) for idx in range(len(self._class_str2int))
            ]

            logger.info(f"Total number of samples/residues: {n_samples}")
            logger.info("Individual class counts and weights:")
            for c in counter:
                logger.info(f"\t{self._class_int2str[c]} : {counter[c]} ({class_weights[c]:.3f})")
            return torch.FloatTensor(class_weights)
        else:
            raise NotImplementedError

    def get_datasets(self, id2emb: Dict[str, Any], output_vars: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        # Parse FASTA protein sequences
        protein_sequences = read_FASTA(self._sequence_file)
        id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}

        self._calculate_id2target(protein_sequences, output_vars)
        self._validate_id2target(id2fasta)

        training_ids, validation_ids, testing_ids = get_split_lists(self._id2attributes)

        if len(training_ids) < 1 or len(validation_ids) < 1 or len(testing_ids) < 1:
            raise ValueError("Not enough samples for training, validation and testing!")

        output_vars['training_ids'] = training_ids
        output_vars['validation_ids'] = validation_ids
        output_vars['testing_ids'] = testing_ids

        train_dataset = get_dataset(self._protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(self._id2target[idx])) for idx in training_ids
        })
        val_dataset = get_dataset(self._protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(self._id2target[idx])) for idx in validation_ids
        })
        test_dataset = get_dataset(self._protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(self._id2target[idx])) for idx in testing_ids
        })
        return train_dataset, val_dataset, test_dataset
