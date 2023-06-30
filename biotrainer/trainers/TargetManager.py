import torch
import logging
import itertools
import numpy as np

from collections import Counter
from Bio.SeqRecord import SeqRecord
from typing import Dict, Any, Tuple, Optional, List

from ..protocols import Protocol
from ..utilities import get_attributes_from_seqrecords, get_attributes_from_seqrecords_for_protein_interactions, \
    get_split_lists, MASK_AND_LABELS_PAD_VALUE, read_FASTA, INTERACTION_INDICATOR, DatasetSample

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

    # Interaction operations
    interaction_operations = {
        "multiply": lambda embedding_left, embedding_right: torch.mul(embedding_left, embedding_right),
        "concat": lambda embedding_left, embedding_right: torch.concat([embedding_left, embedding_right])
    }

    def __init__(self, protocol: Protocol, sequence_file: str,
                 labels_file: Optional[str] = None, mask_file: Optional[str] = None,
                 ignore_file_inconsistencies: Optional[bool] = False,
                 cross_validation_method: str = "",
                 interaction: Optional[str] = None):
        self.protocol = protocol
        self._sequence_file = sequence_file
        self._labels_file = labels_file
        self._mask_file = mask_file
        self._ignore_file_inconsistencies = ignore_file_inconsistencies
        self._cross_validation_method = cross_validation_method
        self._interaction = interaction

    def _calculate_targets(self):
        # Parse FASTA protein sequences
        protein_sequences: List[SeqRecord] = read_FASTA(self._sequence_file)
        id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}
        attributes_from_seqrecords_function = get_attributes_from_seqrecords
        if self._interaction:
            attributes_from_seqrecords_function = get_attributes_from_seqrecords_for_protein_interactions

        # 1. Residue Level
        if self.protocol in Protocol.per_residue_protocols():
            # Expect labels file to be in FASTA format, with each "residue" being the residue-associated-label
            label_sequences = read_FASTA(self._labels_file)
            self._id2attributes = attributes_from_seqrecords_function(label_sequences)

            # Generate Mapping from Ids to Labels
            self._id2target = {label.id: str(label.seq) for label in label_sequences}

            # a) Class output
            if self.protocol in Protocol.classification_protocols():
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
        elif self.protocol in Protocol.per_sequence_protocols():

            # In sequence task, split definitions are in sequence header, as well as target
            # For more info check file specifications!
            self._id2attributes = attributes_from_seqrecords_function(protein_sequences)

            self._id2target = {seq_id: seq_vals["TARGET"] for seq_id, seq_vals in self._id2attributes.items()}
            # a) Class output
            if self.protocol in Protocol.classification_protocols():
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
            elif self.protocol in Protocol.regression_protocols():
                self._id2target = {seq_id: float(seq_val) for seq_id, seq_val in self._id2target.items()}
                self.number_of_outputs = 1
            else:
                raise NotImplementedError
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
        if self._interaction:
            all_protein_ids = []
            for interaction_key in self._id2target.keys():
                protein_ids = interaction_key.split(INTERACTION_INDICATOR)
                all_protein_ids.extend(protein_ids)
            all_ids_with_target = set(all_protein_ids)
        else:
            all_ids_with_target = set(self._id2target.keys())

        for seq_id, seq in id2emb.items():
            # Check that all embeddings have a corresponding label
            if seq_id not in all_ids_with_target:
                embeddings_without_labels.append(seq_id)
            # Make sure the length of the sequences in the embeddings match the length of the seqs in the labels
            elif self.protocol in Protocol.per_residue_protocols() and len(seq) != self._id2target[seq_id].size:
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

    @staticmethod
    def _validate_embeddings_shapes(id2emb: Dict[str, Any]):
        first_embedding_shape = list(id2emb.values())[0].shape[-1]  # Last position in shape is always embedding length
        all_embeddings_have_same_dimension = all([embedding.shape[-1] == first_embedding_shape
                                                  for embedding in id2emb.values()])
        if not all_embeddings_have_same_dimension:
            raise Exception(f"Embeddings dimensions differ between sequences, but all must be equal!")

    def get_datasets_by_annotations(self, id2emb: Dict[str, Any]) -> \
            Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
        # At first calculate id2target and validate
        self._calculate_targets()
        self._validate_targets(id2emb)

        # Get dataset splits from file
        self.training_ids, self.validation_ids, self.testing_ids = get_split_lists(self._id2attributes)

        # Check dataset splits are not empty
        def except_on_empty(split_ids: List[str], name: str):
            if len(split_ids) == 0:
                raise Exception(f"The provided {name} set is empty! Please provide at least one sequence for "
                                f"the {name} set.")

        if not self._ignore_file_inconsistencies:
            except_on_empty(split_ids=self.training_ids, name="training")
            if self._cross_validation_method == "hold_out":
                except_on_empty(split_ids=self.validation_ids, name="validation")
            except_on_empty(split_ids=self.testing_ids, name="test")

        # Combine embeddings for protein_protein_interaction
        if self._interaction:
            interaction_operation = self.interaction_operations.get(self._interaction)
            if not interaction_operation:
                raise NotImplementedError(f"Chosen interaction operation {self._interaction} is not supported!")
            interaction_id2emb = {}
            for interaction_id in self._id2attributes.keys():
                interactor_left = interaction_id.split(INTERACTION_INDICATOR)[0]
                interactor_right = interaction_id.split(INTERACTION_INDICATOR)[1]
                embedding_left = id2emb[interactor_left]
                embedding_right = id2emb[interactor_right]
                combined_embedding = interaction_operation(embedding_left, embedding_right)
                interaction_id2emb[interaction_id] = combined_embedding

            id2emb = interaction_id2emb

        # Validate that all embeddings have the same shape
        self._validate_embeddings_shapes(id2emb)

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
        if self.protocol in Protocol.classification_protocols():
            training_targets = [self._id2target[training_id] for training_id in self.training_ids]
            if self.protocol in Protocol.per_residue_protocols():
                # concatenate all targets irrespective of protein to count class sizes
                training_targets = list(itertools.chain.
                                        from_iterable([list(targets) for targets in training_targets]))
                if self._mask_file:
                    # Ignore unresolved residues for class weights
                    training_targets = [target for target in training_targets if target != MASK_AND_LABELS_PAD_VALUE]

                counter = Counter(training_targets)
            else:  # Per-sequence
                counter = Counter([target.item() for target in training_targets])
            # total number of samples in the set irrespective of classes
            n_samples = sum([counter[idx] for idx in range(len(self.class_str2int))])
            n_classes = len(counter.keys())
            # balanced class weighting (inversely proportional to class size)
            class_weights = [(n_samples / (n_classes * counter[idx])) if counter[idx] > 0 else 1
                             for idx in range(len(self.class_str2int))]

            logger.info(f"Total number of sequences/residues: {n_samples}")
            logger.info("Individual class counts and weights (training set):")
            for idx in range(len(self.class_str2int)):
                logger.info(f"\t{self.class_int2str[idx]} : {counter[idx]} ({class_weights[idx]:.3f})")
            if n_classes < len(self.class_str2int):
                missing_classes = [self.class_int2str[idx] for idx in range(len(self.class_str2int))
                                   if counter[idx] == 0]
                logger.warning(f"The training set does not contain samples "
                               f"for the following classes: {missing_classes}!")
            return torch.FloatTensor(class_weights)
        else:
            raise Exception(f"Class weights can only be calculated for classification tasks!")
