import torch
import logging
import itertools
import numpy as np

from collections import Counter
from Bio.SeqRecord import SeqRecord
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, Optional, List

from ..datasets import get_dataset
from .target_manager_utils import get_split_lists
from ..utilities import read_FASTA, get_attributes_from_seqrecords, MASK_AND_LABELS_PAD_VALUE

logger = logging.getLogger(__name__)


class TargetManager:
    _id2target: Dict[str, Any] = dict()
    _id2attributes: Dict[str, Any] = dict()
    # This will be 1 for regression tasks, 2 for binary classification tasks, and N>2 for everything else
    number_of_outputs: int = 1

    # Optional, must be set in _calculate_targets()
    class_str2int: Optional[Dict[str, int]] = None
    class_int2str: Optional[Dict[int, str]] = None
    _class_labels: Optional[List[str]] = None

    # Dataset split lists
    training_ids = None
    validation_ids = None
    testing_ids = None

    def __init__(self, protocol: str, sequence_file: str,
                 labels_file: Optional[str] = None, mask_file: Optional[str] = None):
        self._protocol = protocol
        self._sequence_file = sequence_file
        self._labels_file = labels_file
        self._mask_file = mask_file

    def _calculate_targets(self):
        # Parse FASTA protein sequences
        protein_sequences: List[SeqRecord] = read_FASTA(self._sequence_file)
        id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}

        # 1. Residue Level
        if 'residue_' in self._protocol:
            # Expect labels file to be in FASTA format, with each "residue" being the residue-associated-label
            label_sequences = read_FASTA(self._labels_file)
            self._id2attributes = get_attributes_from_seqrecords(label_sequences)

            # Generate Mapping from Ids to Labels
            self._id2target = {label.id: str(label.seq) for label in label_sequences}

            # a) Class output
            if 'class' in self._protocol:
                class_labels_temp = set()

                # Infer classes from data
                for classes in self._id2target.values():
                    class_labels_temp = class_labels_temp | set(classes)
                self._class_labels = sorted(class_labels_temp)

                self.number_of_outputs = len(self._class_labels)

                # Create a mapping from integers to class labels and reverse
                self.class_str2int = {letter: idx for idx, letter in enumerate(self._class_labels)}
                self.class_int2str = {idx: letter for idx, letter in enumerate(self._class_labels)}

                # Convert label values to lists of numbers based on the maps
                self._id2target = {identifier: np.array([self.class_str2int[label] for label in labels])
                                   for identifier, labels in self._id2target.items()}  # classes idxs (zero-based)
                # Apply masks if provided:
                if self._mask_file:
                    sequence_masks = read_FASTA(self._mask_file)
                    mask2fasta = {protein.id: np.array([int(mask_value) for mask_value in str(protein.seq)])
                                  for protein in sequence_masks}
                    for identifier, unmasked in self._id2target.items():
                        mask = mask2fasta[identifier]
                        target_with_mask = np.array([value if mask[index] == 1 else MASK_AND_LABELS_PAD_VALUE for
                                                     index, value in enumerate(unmasked)])
                        self._id2target[identifier] = target_with_mask
            # b) Value output
            else:
                raise NotImplementedError

        # 2. Sequence Level
        elif 'sequence_' in self._protocol or 'residues_' in self._protocol:

            # In sequence task, split definitions are in sequence header, as well as target
            # For more info check file specifications!
            self._id2attributes = get_attributes_from_seqrecords(protein_sequences)

            self._id2target = {seq_id: seq_vals["TARGET"] for seq_id, seq_vals in self._id2attributes.items()}
            # a) Class output
            if 'class' in self._protocol:
                # Infer classes from data
                self._class_labels = sorted(set(self._id2target.values()))
                self.number_of_outputs = len(self._class_labels)

                # Create a mapping from integers to class labels and reverse
                self.class_str2int = {letter: idx for idx, letter in enumerate(self._class_labels)}
                self.class_int2str = {idx: letter for idx, letter in enumerate(self._class_labels)}

                # Convert label values to lists of numbers based on the maps
                self._id2target = {identifier: np.array(self.class_str2int[label])
                                   for identifier, label in self._id2target.items()}  # classes idxs (zero-based)

            # b) Value output
            elif 'value' in self._protocol:
                self._id2target = {seq_id: float(seq_val) for seq_id, seq_val in self._id2target.items()}
                self.number_of_outputs = 1
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        if not self._id2target:
            raise Exception("Prediction targets not found or could not be extracted!")

        self._validate_targets(id2fasta)

    def _validate_targets(self, id2fasta):
        if 'residue_' in self._protocol:
            invalid_sequence_lengths = []
            for seq_id, seq in id2fasta.items():
                # Check that all sequences are included in both the sequence and labels file
                if seq_id not in self._id2target.keys():
                    raise Exception(f"Sequence {seq_id} not found in labels file! Make sure that all your sequences "
                                    f"are present and annotated in the labels file.")
                # Make sure the length of the sequences in the FASTA matches the length of the sequences in the labels
                if len(seq) != self._id2target[seq_id].size:
                    invalid_sequence_lengths.append((seq_id, len(seq), self._id2target[seq_id].size))

            if len(invalid_sequence_lengths) > 0:
                exception_message = f"Length mismatch for {len(invalid_sequence_lengths)} sequence(s)!\n"
                for seq_id, seq_len, target_len in invalid_sequence_lengths:
                    exception_message += f"{seq_id}: Sequence={seq_len} vs. Labels={self._id2target[seq_id].size}\n"
                raise Exception(exception_message[:-1])  # Discard last \n

    def get_datasets(self, id2emb: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        # At first calculate id2target
        self._calculate_targets()

        # Get dataset splits from file
        self.training_ids, self.validation_ids, self.testing_ids = get_split_lists(self._id2attributes)

        # Create datasets
        train_dataset = get_dataset(self._protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(self._id2target[idx])) for idx in self.training_ids
        })
        val_dataset = get_dataset(self._protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(self._id2target[idx])) for idx in self.validation_ids
        })
        test_dataset = get_dataset(self._protocol, {
            idx: (torch.tensor(id2emb[idx]), torch.tensor(self._id2target[idx])) for idx in self.testing_ids
        })

        return train_dataset, val_dataset, test_dataset

    def compute_class_weights(self) -> torch.FloatTensor:
        if 'class' in self._protocol:
            # concatenate all labels irrespective of protein to count class sizes
            counter = Counter(list(itertools.chain.from_iterable(
                [list(labels) for labels in self._id2target.values()]
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
            return torch.FloatTensor(class_weights)
        else:
            raise Exception(f"Class weights can only be calculated for classification tasks!")

    def revert_mappings(self, test_predictions: List) -> List:
        # If residue-to-class problem, map the integers back to the class labels (single letters)
        if self._protocol == 'residue_to_class':
            return ["".join(
                [self.class_int2str[p] for p in prediction]
            ) for prediction in test_predictions]

        # If sequence/residues-to-class problem, map the integers back to the class labels (whatever length)
        elif self._protocol == "sequence_to_class" or self._protocol == "residues_to_class":
            return [self.class_int2str[p] for p in test_predictions]
        else:
            return test_predictions
