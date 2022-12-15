import torch
import random
import logging
import itertools
import numpy as np

from collections import Counter
from Bio.SeqRecord import SeqRecord
from typing import Dict, Any, Tuple, Optional, List

from .target_manager_utils import get_split_lists

from ..utilities import get_attributes_from_seqrecords, get_attributes_from_seqrecords_for_protein_interactions, \
    MASK_AND_LABELS_PAD_VALUE, read_FASTA, DatasetSample

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
    training_ids: List[str] = None
    validation_ids: List[str] = None
    testing_ids: List[str] = None

    def __init__(self, protocol: str, sequence_file: str,
                 labels_file: Optional[str] = None, mask_file: Optional[str] = None,
                 ignore_file_inconsistencies: Optional[bool] = False,
                 limited_sample_size: Optional[int] = -1):
        self.protocol = protocol
        self._sequence_file = sequence_file
        self._labels_file = labels_file
        self._mask_file = mask_file
        self._ignore_file_inconsistencies = ignore_file_inconsistencies
        self._limited_sample_size = limited_sample_size

    def _calculate_targets(self):
        # Parse FASTA protein sequences
        protein_sequences: List[SeqRecord] = read_FASTA(self._sequence_file)
        id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}

        # 1. Residue Level
        if 'residue_' in self.protocol:
            # Expect labels file to be in FASTA format, with each "residue" being the residue-associated-label
            label_sequences = read_FASTA(self._labels_file)
            self._id2attributes = get_attributes_from_seqrecords(label_sequences)

            # Generate Mapping from Ids to Labels
            self._id2target = {label.id: str(label.seq) for label in label_sequences}

            # a) Class output
            if 'class' in self.protocol:
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
        elif 'sequence_' in self.protocol or 'residues_' in self.protocol:

            # In sequence task, split definitions are in sequence header, as well as target
            # For more info check file specifications!
            self._id2attributes = get_attributes_from_seqrecords(protein_sequences)

            self._id2target = {seq_id: seq_vals["TARGET"] for seq_id, seq_vals in self._id2attributes.items()}
            # a) Class output
            if 'class' in self.protocol:
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
            elif 'value' in self.protocol:
                self._id2target = {seq_id: float(seq_val) for seq_id, seq_val in self._id2target.items()}
                self.number_of_outputs = 1
            else:
                raise NotImplementedError

        # Protein-Protein Interactions
        elif 'protein_protein_interaction' in self.protocol:
            self._id2attributes = get_attributes_from_seqrecords_for_protein_interactions(protein_sequences)

            self._id2target = {interaction_id: seq_vals["TARGET"] for interaction_id, seq_vals
                               in self._id2attributes.items()}

            # a) Class output
            # if 'class' in self.protocol:
            # Infer classes from data
            self._class_labels = sorted(set(self._id2target.values()))
            self.number_of_outputs = len(self._class_labels)

            # Create a mapping from integers to class labels and reverse
            self.class_str2int = {letter: idx for idx, letter in enumerate(self._class_labels)}
            self.class_int2str = {idx: letter for idx, letter in enumerate(self._class_labels)}

            # Convert label values to lists of numbers based on the maps
            self._id2target = {identifier: np.array(self.class_str2int[label])
                               for identifier, label in self._id2target.items()}  # classes idxs (zero-based)

            """
            # b) Value output
            elif 'value' in self.protocol:
                self._id2target = {seq_id: float(seq_val) for seq_id, seq_val in self._id2target.items()}
                self.number_of_outputs = 1
            else:
                raise NotImplementedError
            """
        else:
            raise NotImplementedError

        if not self._id2target:
            raise Exception("Prediction targets not found or could not be extracted!")

    def _validate_targets(self, id2emb: Dict[str, Any]):
        """
        1. Check if number of embeddings and corresponding labels match for every protocol:
            # embeddings == # labels --> SUCCESS
            # embeddings > # labels --> Fail, unless self._ignore_file_inconsistencies (=> drop embeddings).
            # embeddings < # labels --> Fail, unless self._ignore_file_inconsistencies (=> drop labels).
        2. Check that lengths of embeddings == length of provided labels, if not --> Fail. (only residue_to_x)
        """
        invalid_sequence_lengths = []
        embeddings_without_labels = []
        labels_without_embeddings = []
        if "_interaction" in self.protocol:
            all_protein_ids = []
            for interaction_key in self._id2target.keys():
                protein_ids = interaction_key.split("§")
                all_protein_ids.extend(protein_ids)
            all_ids_with_target = set(all_protein_ids)
        else:
            all_ids_with_target = set(self._id2target.keys())

        for seq_id, seq in id2emb.items():
            # Check that all embeddings have a corresponding label
            if seq_id not in all_ids_with_target:
                embeddings_without_labels.append(seq_id)
            # Make sure the length of the sequences in the embeddings match the length of the seqs in the labels
            elif "residue_" in self.protocol and len(seq) != self._id2target[seq_id].size:
                invalid_sequence_lengths.append((seq_id, len(seq), self._id2target[seq_id].size))

        for seq_id in all_ids_with_target:
            # Check that all labels have a corresponding embedding
            if seq_id not in id2emb.keys():
                labels_without_embeddings.append(seq_id)

        if len(embeddings_without_labels) > 0:
            if self._ignore_file_inconsistencies:
                logger.warning(f"Found {len(embeddings_without_labels)} embedding(s) without a corresponding "
                               f"entry in the labels file! Because ignore_file_inconsistencies flag is set, "
                               f"these sequences are dropped for training. "
                               f"Data loss: {(len(embeddings_without_labels) / len(id2emb.keys())) * 100:.2f}%")
                for seq_id in embeddings_without_labels:
                    id2emb.pop(seq_id)  # Remove redundant sequences
            else:
                exception_message = f"{len(embeddings_without_labels)} sequence(s) not found in labels file! " \
                                    f"Make sure that all sequences are present and annotated in the labels file " \
                                    f"or set the ignore_file_inconsistencies flag to True.\n" \
                                    f"Missing label sequence ids:\n"
                for seq_id in embeddings_without_labels:
                    exception_message += f"Sequence {seq_id}\n"
                raise Exception(exception_message[:-1])  # Discard last \n

        if len(labels_without_embeddings) > 0:
            if self._ignore_file_inconsistencies:
                logger.warning(f"Found {len(labels_without_embeddings)} label(s) without a corresponding "
                               f"entry in the embeddings file! Because ignore_file_inconsistencies flag is set, "
                               f"these labels are dropped for training. "
                               f"Data loss: {(len(labels_without_embeddings) / len(id2emb.keys())) * 100:.2f}%")
                for seq_id in labels_without_embeddings:
                    self._id2target.pop(seq_id)  # Remove redundant labels
                    self._id2attributes.pop(seq_id)  # Remove redundant labels
            else:
                exception_message = f"{len(labels_without_embeddings)} label(s) not found in embeddings file! " \
                                    f"Make sure that for every sequence id in the labels file, there is a " \
                                    f"corresponding embedding. If no pre-computed embeddings were used, \n" \
                                    f"this error message indicates that there is a sequence missing in the " \
                                    f"sequence_file.\n" \
                                    f"Setting the ignore_file_inconsistencies flag to True " \
                                    f"will ignore this problem.\n" \
                                    f"Missing sequence ids:\n"
                for seq_id in labels_without_embeddings:
                    exception_message += f"Sequence {seq_id}\n"
                raise Exception(exception_message[:-1])  # Discard last \n

        if len(invalid_sequence_lengths) > 0:
            exception_message = f"Length mismatch for {len(invalid_sequence_lengths)} sequence(s)!\n"
            for seq_id, seq_len, target_len in invalid_sequence_lengths:
                exception_message += f"{seq_id}: Sequence={seq_len} vs. Labels={self._id2target[seq_id].size}\n"
            raise Exception(exception_message[:-1])  # Discard last \n

    def get_datasets_by_annotations(self, id2emb: Dict[str, Any]) -> \
            Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
        # At first calculate id2target and validate
        self._calculate_targets()
        self._validate_targets(id2emb)

        # Get dataset splits from file
        self.training_ids, self.validation_ids, self.testing_ids = get_split_lists(self._id2attributes)
        # Apply limited sample number
        if self._limited_sample_size > 0:
            logger.info(f"Using limited sample size of {self._limited_sample_size} for train, validation, test")
            self.training_ids = random.sample(self.training_ids,
                                              k=min(self._limited_sample_size, len(self.training_ids)))
            """  TODO: Also on val/test ids?
            self.validation_ids = random.sample(self.validation_ids,
                                                k=min(self._limited_sample_size, len(self.validation_ids)))
            self.testing_ids = random.sample(self.testing_ids,
                                             k=min(self._limited_sample_size, len(self.testing_ids)))
            """
            remaining_ids = self.training_ids + self.validation_ids + self.testing_ids
            self._id2target = {idx: target for idx, target in self._id2target.items() if idx in remaining_ids}

        # Concatenate embeddings for protein_protein_interaction
        if "protein_protein_interaction" in self.protocol:
            for interaction_id in self._id2attributes.keys():
                interactor_left = interaction_id.split("§")[0]
                interactor_right = interaction_id.split("§")[1]
                embedding_left = id2emb[interactor_left]
                embedding_right = id2emb[interactor_right]
                combined_embedding = torch.mul(embedding_left, embedding_right)
                id2emb[interaction_id] = combined_embedding

        # Create datasets
        train_dataset = [
            DatasetSample(idx, id2emb[idx], torch.tensor(self._id2target[idx])) for idx in self.training_ids
        ]
        val_dataset = [
            DatasetSample(idx, id2emb[idx], torch.tensor(self._id2target[idx])) for idx in self.validation_ids
        ]
        test_dataset = [
            DatasetSample(idx, id2emb[idx], torch.tensor(self._id2target[idx])) for idx in self.testing_ids
        ]

        return train_dataset, val_dataset, test_dataset

    def compute_class_weights(self) -> torch.FloatTensor:
        if 'class' in self.protocol or "_interaction" in self.protocol:
            # concatenate all labels irrespective of protein to count class sizes
            if "residue_" in self.protocol:
                counter = Counter(list(itertools.chain.from_iterable(
                    [list(labels) for labels in self._id2target.values()]
                )))
            else:
                counter = Counter([label.item() for label in self._id2target.values()])
            # total number of samples in the set irrespective of classes
            n_samples = sum([counter[idx] for idx in range(len(self.class_str2int))])
            # balanced class weighting (inversely proportional to class size)
            class_weights = [
                (n_samples / (len(self.class_str2int) * counter[idx])) for idx in range(len(self.class_str2int))
            ]

            logger.info(f"Total number of sequences/residues: {n_samples}")
            logger.info("Individual class counts and weights:")
            for c in counter:
                logger.info(f"\t{self.class_int2str[c]} : {counter[c]} ({class_weights[c]:.3f})")
            return torch.FloatTensor(class_weights)
        else:
            raise Exception(f"Class weights can only be calculated for classification tasks!")
