import torch
import itertools

from pathlib import Path
from collections import Counter
from typing import Dict, Any, Tuple, Optional, List, Union

from ..protocols import Protocol
from ..input_files import BiotrainerSequenceRecord, merge_protein_interactions, get_split_lists, read_FASTA
from ..utilities import MASK_AND_LABELS_PAD_VALUE, INTERACTION_INDICATOR, EmbeddingDatasetSample, get_logger, \
    SequenceDatasetSample

logger = get_logger(__name__)


class TargetManager:
    _input_records: Dict[str, BiotrainerSequenceRecord]  # Hash To Record (for Fine-Tuning)
    _input_ids: Dict[str, List[str]] = dict()  # Hash To List of IDs
    _id2target: Dict[str, Any] = dict()  # Hash To Target
    _id2sets: Dict[str, str] = dict()  # Hash to Set
    # This will be 1 for regression tasks, 2 for binary classification tasks, and N>2 for everything else
    number_of_outputs: int = 1

    # Optional for classification tasks, must be set in _calculate_targets()
    class_str2int: Optional[Dict[str, int]] = None
    class_int2str: Optional[Dict[int, str]] = None
    _class_labels: Optional[List[str]] = None

    # Dataset split lists
    training_ids: List[str] = None
    validation_ids: List[str] = None
    testing_ids: Dict[str, List[str]] = None
    prediction_ids: List[str] = None

    # Interaction operations
    interaction_operations = {
        "multiply": lambda embedding_left, embedding_right: torch.mul(embedding_left, embedding_right),
        "concat": lambda embedding_left, embedding_right: torch.concat([embedding_left, embedding_right])
    }

    def __init__(self, protocol: Protocol, input_data: Union[str, Path, List[BiotrainerSequenceRecord]],
                 ignore_file_inconsistencies: Optional[bool] = False,
                 cross_validation_method: str = "",
                 interaction: Optional[str] = None):
        self.protocol = protocol
        self._input_data = input_data
        self._ignore_file_inconsistencies = ignore_file_inconsistencies
        self._cross_validation_method = cross_validation_method
        self._interaction = interaction
        self._loaded = False

    def load(self):
        self._calculate_targets()

        # Get dataset splits from file
        self.training_ids, self.validation_ids, self.testing_ids, self.prediction_ids = get_split_lists(self._id2sets)

        # Check dataset splits are not empty
        def except_on_empty(split_ids: List[str], name: str):
            if len(split_ids) == 0:
                raise ValueError(f"The provided {name} set is empty! Please provide at least one sequence for "
                                 f"the {name} set.")

        if not self._ignore_file_inconsistencies:
            except_on_empty(split_ids=self.training_ids, name="training")
            if self._cross_validation_method == "hold_out":
                except_on_empty(split_ids=self.validation_ids, name="validation")
            for test_set in self.testing_ids.values():
                except_on_empty(split_ids=test_set, name="test")

        self._loaded = True

    def _calculate_targets(self):
        # Parse FASTA protein sequences if not done yet
        if isinstance(self._input_data, str) or isinstance(self._input_data, Path):
            input_records: List[BiotrainerSequenceRecord] = read_FASTA(self._input_data)
        else:
            input_records: List[BiotrainerSequenceRecord] = self._input_data
        assert isinstance(input_records[0], BiotrainerSequenceRecord)

        # Store records for fine-tuning sequence datasets
        self._input_records = {seq_record.get_hash(): seq_record for seq_record in input_records}

        # Store input ids for better error messages
        for seq_record in input_records:
            seq_hash = seq_record.get_hash()
            if seq_hash not in self._input_ids:
                self._input_ids[seq_hash] = []
            self._input_ids[seq_hash].append(seq_record.seq_id)

        # id2X => sequence hash to X
        self._id2target, id2masks, self._id2sets = BiotrainerSequenceRecord.get_dicts(input_records)
        assert len(self._id2target.keys()) == len(id2masks.keys()) \
               == len(self._id2sets.keys()), f"Length mismatch after reading input file!"

        if self._interaction:
            merged_interactions_dict = merge_protein_interactions(input_records)
            self._id2target = {int_id: int_dict["TARGET"] for int_id, int_dict in
                               merged_interactions_dict.items()}
            self._id2sets = {int_id: int_dict["SET"] for int_id, int_dict in
                             merged_interactions_dict.items()}

        # Remove None targets from pred set
        self._id2target = {seq_hash: target for seq_hash, target in self._id2target.items() if target is not None}
        if len(self._id2target) == 0:
            raise ValueError("Could not parse any valid targets from given input file!")

        # 1. Residue Level
        if self.protocol in Protocol.per_residue_protocols():
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
                self._id2target = {identifier: torch.tensor([self.class_str2int[label] for label in labels]).long()
                                   for identifier, labels in self._id2target.items()}  # classes idxs (zero-based)

                # MASKS
                self._apply_masks(id2masks=id2masks)
            # b) Value output
            elif self.protocol in Protocol.regression_protocols():
                self._id2target = {seq_hash: torch.tensor(res_val).float()
                                   for seq_hash, res_val in self._id2target.items()}
                self.number_of_outputs = 1
                self._apply_masks(id2masks=id2masks)
            else:
                raise NotImplementedError

        # 2. Sequence Level
        elif self.protocol in Protocol.per_sequence_protocols():
            # a) Class output
            if self.protocol in Protocol.classification_protocols():
                # Infer classes from data
                self._class_labels = sorted(set(self._id2target.values()))
                self.number_of_outputs = len(self._class_labels)

                # Create a mapping from integers to class labels and reverse
                self.class_str2int = {letter: idx for idx, letter in enumerate(self._class_labels)}
                self.class_int2str = {idx: letter for idx, letter in enumerate(self._class_labels)}

                # Convert label values to lists of numbers based on the maps
                self._id2target = {seq_hash: torch.tensor(self.class_str2int[label]).long()
                                   for seq_hash, label in self._id2target.items()}  # classes idxs (zero-based)

            # b) Value output
            elif self.protocol in Protocol.regression_protocols():
                self._id2target = {seq_hash: torch.tensor(float(seq_val)).float()
                                   for seq_hash, seq_val in self._id2target.items()}
                self.number_of_outputs = 1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Double-checking after target loading for correctness
        if self.protocol in Protocol.regression_protocols():
            assert self.number_of_outputs == 1, f"Number of outputs does not equal 1 for regression protocol!"
        if not self._id2target or len(self._id2target) == 0:
            raise ValueError("Prediction targets not found or could not be extracted!")

    def _apply_masks(self, id2masks):
        # Validate masks (each mask must contain at least one resolved (1) value)
        for seq_hash, mask in id2masks.items():
            if mask is not None and 1 not in mask:
                raise ValueError(f"{self._input_ids[seq_hash]} does not have a valid mask as it does "
                                 f"not contain at least one resolved (1) value!")

        # Replace labels with masking value
        for seq_hash, unmasked_target in self._id2target.items():
            mask = id2masks[seq_hash]
            if mask is not None:
                target_with_mask = torch.tensor([value if mask[index] == 1 else MASK_AND_LABELS_PAD_VALUE for
                                                 index, value in enumerate(unmasked_target)])
                target_with_mask = target_with_mask.long() if self.protocol in Protocol.classification_protocols() \
                    else target_with_mask.float()
                self._id2target[seq_hash] = target_with_mask

    def _validate_targets(self, id2emb: Dict[str, torch.tensor]):
        """
        1. Check if number of embeddings and corresponding labels match for every protocol:
            # embeddings == # labels --> SUCCESS
            # embeddings > # labels --> Fail, unless self._ignore_file_inconsistencies (=> drop embeddings).
            # embeddings < # labels --> Fail, unless self._ignore_file_inconsistencies (=> drop labels).
        2. Check that lengths of embeddings == length of provided labels, if not --> Fail. (only residue_to_x)
        """
        invalid_embeddings_lengths = []
        embeddings_without_labels = []
        labels_without_embeddings = []
        if self._interaction:
            all_protein_ids = []
            for interaction_key in self._id2target.keys():
                protein_ids = interaction_key.split(INTERACTION_INDICATOR)
                all_protein_ids.extend(protein_ids)
            all_hashes_with_target = set(all_protein_ids)
        else:
            all_hashes_with_target = set(self._id2target.keys())

        for seq_hash, embd in id2emb.items():
            # Check that all embeddings have a corresponding label
            if seq_hash in self.prediction_ids:
                continue

            if seq_hash not in all_hashes_with_target:
                embeddings_without_labels.append(seq_hash)
            # Make sure the length of the sequences in the embeddings match the length of the seqs in the labels
            elif self.protocol in Protocol.per_residue_protocols() and len(embd) != self._id2target[seq_hash].size()[0]:
                invalid_embeddings_lengths.append((seq_hash, len(embd), self._id2target[seq_hash].size()))

        for seq_hash in all_hashes_with_target:
            # Check that all labels have a corresponding embedding
            if seq_hash not in id2emb.keys():
                labels_without_embeddings.append(seq_hash)

        if len(embeddings_without_labels) > 0:
            if self._ignore_file_inconsistencies:
                if len(embeddings_without_labels) == len(id2emb):
                    raise ValueError(f"Did not find any labels for any given embedding!")

                logger.warning(f"Found {len(embeddings_without_labels)} embedding(s) without a corresponding "
                               f"entry in the labels file! Because ignore_file_inconsistencies flag is set, "
                               f"these sequences are dropped for training. "
                               f"Data loss: {(len(embeddings_without_labels) / len(id2emb.keys())) * 100:.2f}%")
                for seq_hash in embeddings_without_labels:
                    id2emb.pop(seq_hash)  # Remove redundant sequences
            else:
                exception_message = f"{len(embeddings_without_labels)} sequence(s) not found in labels file! " \
                                    f"Make sure that all sequences are present and annotated in the labels file " \
                                    f"or set the ignore_file_inconsistencies flag to True.\n" \
                                    f"Missing label sequence ids:\n"
                for seq_hash in embeddings_without_labels:
                    exception_message += f"Sequence - ID(s): {self._input_ids.get(seq_hash, 'N/A')} - Hash: {seq_hash}\n"
                raise ValueError(exception_message[:-1])  # Discard last \n

        if len(labels_without_embeddings) > 0:

            if self._ignore_file_inconsistencies:
                logger.warning(f"Found {len(labels_without_embeddings)} label(s) without a corresponding "
                               f"entry in the embeddings file! Because ignore_file_inconsistencies flag is set, "
                               f"these labels are dropped for training. "
                               f"Data loss: {(len(labels_without_embeddings) / len(id2emb.keys())) * 100:.2f}%")
                for seq_hash in labels_without_embeddings:
                    self._id2target.pop(seq_hash)  # Remove redundant labels
            else:
                exception_message = f"{len(labels_without_embeddings)} label(s) not found in embeddings file! " \
                                    f"Make sure that for every sequence id in the labels file, there is a " \
                                    f"corresponding embedding. If no pre-computed embeddings were used, \n" \
                                    f"this error message indicates that there is a sequence missing in the " \
                                    f"sequence_file.\n" \
                                    f"Setting the ignore_file_inconsistencies flag to True " \
                                    f"will ignore this problem.\n" \
                                    f"Missing sequence ids:\n"
                for seq_hash in labels_without_embeddings:
                    exception_message += f"Sequence - ID(s): {self._input_ids.get(seq_hash, 'N/A')} - Hash: {seq_hash}\n"
                raise ValueError(exception_message[:-1])  # Discard last \n

        if len(invalid_embeddings_lengths) > 0:
            exception_message = f"Length mismatch for {len(invalid_embeddings_lengths)} embedding(s)!\n"
            for seq_hash, seq_len, target_len in invalid_embeddings_lengths:
                exception_message += (
                    f"Embedding - ID(s): {self._input_ids.get(seq_hash, 'N/A')} - Sequence Hash: {seq_hash} - "
                    f"Embedding_Length={seq_len} vs. Labels_Length={self._id2target[seq_hash].size()[0]}\n")
            raise ValueError(exception_message[:-1])  # Discard last \n

    @staticmethod
    def _validate_embeddings_shapes(id2emb: Dict[str, Any]):
        shapes = set([val.shape[-1] for val in id2emb.values()])
        all_embeddings_have_same_dimension = len(shapes) == 1
        if not all_embeddings_have_same_dimension:
            raise ValueError(f"Embeddings dimensions differ between sequences, but all must be equal!\n"
                             f"Found: {shapes}")

    def get_embedding_datasets(self, id2emb: Dict[str, Any]) -> \
            Tuple[List[EmbeddingDatasetSample], List[EmbeddingDatasetSample], Dict[str, List[EmbeddingDatasetSample]],
            List[EmbeddingDatasetSample]]:
        assert self._loaded, f"Dataset creation called before loading!"
        self._validate_targets(id2emb)
        # Combine embeddings for protein_protein_interaction
        if self._interaction:
            interaction_operation = self.interaction_operations.get(self._interaction)
            if not interaction_operation:
                raise NotImplementedError(f"Chosen interaction operation {self._interaction} is not supported!")
            interaction_id2emb = {}
            for interaction_id in self._id2target.keys():
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
            EmbeddingDatasetSample(idx, id2emb[idx], torch.tensor(self._id2target[idx])) for idx in self.training_ids
        ]
        val_dataset = [
            EmbeddingDatasetSample(idx, id2emb[idx], torch.tensor(self._id2target[idx])) for idx in self.validation_ids
        ]

        test_datasets = {}
        for test_set_id, test_set in self.testing_ids.items():
            test_datasets[test_set_id] = [EmbeddingDatasetSample(idx, id2emb[idx], torch.tensor(self._id2target[idx]))
                                          for idx in test_set]

        pred_dataset = [
            EmbeddingDatasetSample(idx, id2emb[idx], torch.empty(1)) for idx in self.prediction_ids
        ]

        return train_dataset, val_dataset, test_datasets, pred_dataset

    def get_sequence_datasets(self):
        assert self._loaded, f"Dataset creation called before loading!"
        # Create datasets
        train_dataset = [SequenceDatasetSample(idx, self._input_records[idx], self._id2target[idx])
                         for idx in self.training_ids]
        val_dataset = [SequenceDatasetSample(idx, self._input_records[idx], self._id2target[idx])
                       for idx in self.validation_ids]

        test_datasets = {}
        for test_set_id, test_set in self.testing_ids.items():
            test_datasets[test_set_id] = [SequenceDatasetSample(idx, self._input_records[idx], self._id2target[idx])
                                          for idx in test_set]

        pred_dataset = [SequenceDatasetSample(idx, self._input_records[idx], self._id2target[idx])
                        for idx in self.prediction_ids]

        return train_dataset, val_dataset, test_datasets, pred_dataset

    def compute_class_weights(self) -> torch.FloatTensor:
        if self.protocol in Protocol.classification_protocols():
            training_targets = [self._id2target[training_id] for training_id in self.training_ids]
            if self.protocol in Protocol.per_residue_protocols():
                # concatenate all targets irrespective of protein to count class sizes
                training_targets = list(itertools.chain.
                                        from_iterable([list(targets) for targets in training_targets]))
                # Ignore unresolved residues for class weights
                training_targets = [target.item() for target in training_targets if target != MASK_AND_LABELS_PAD_VALUE]

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
            raise ValueError(f"Class weights can only be calculated for classification tasks!")
