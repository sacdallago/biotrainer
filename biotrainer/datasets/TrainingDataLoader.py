import itertools
import logging
import torch
import h5py
import time

import numpy as np

from collections import Counter
from pathlib import Path
from typing import Union, List, Dict, Tuple, Set

from bio_embeddings.utilities.pipeline import execute_pipeline_from_config

from ..utilities import get_device, read_FASTA, attributes_from_seqrecords

logger = logging.getLogger(__name__)


class TrainingDataLoader:

    # The set of classes
    _class_labels: Set

    # A mapping from an integer to a class label (e.g. 0 -> "L")
    _class_str2int: Dict[int, str]

    # A mapping from a class label to an integer (e.g. "L" -> 0)
    _class_int2str: Dict[str, int]

    # A mapping from an index (retrieved from FASTA) to a protein sequence
    _id2fasta: Dict[str, str]

    # A mapping from an index (retrieved from FASTA) to a list of integers
    _id2label: Dict[str, List[int]]

    # A mapping from an index (retrieved from the FASTA or from the h5) to the sequence' embedding
    _id2emb: Dict[str, np.ndarray]

    # Split id store
    _training_ids: List[str]
    _testing_ids: List[str]
    _validation_ids: List[str]

    def __init__(
            self,
            sequence_file: str,
            labels_file: str,
            embedder_name: Union[None, str] = None,
            embeddings_file_path: Union[None, str] = None,
            device: Union[None, str, torch.device] = None
    ):
        self._sequence_file = sequence_file
        self._labels_file = labels_file
        self._device = get_device(device)

        # Creates containers for FASTA sequences and associated labels
        # Also infer classes and create lookup maps to and from class labels
        self._parse_FASTA_files()

        # Load or compute embeddings
        self._retrieve_embeddings(embedder_name=embedder_name, embeddings_file_path=embeddings_file_path)

    def _parse_FASTA_files(self):
        protein_sequences = read_FASTA(self._sequence_file)
        id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}

        label_sequences = read_FASTA(self._labels_file)
        id2label = {label.id: str(label.seq) for label in label_sequences}
        label_attributes = attributes_from_seqrecords(label_sequences)

        # Sanity check: labels must contain SET and VALIDATION attributes
        self._training_ids = list()
        self._validation_ids = list()
        self._testing_ids = list()

        for idx in id2label.keys():
            split = label_attributes[idx].get("SET")

            if split == 'train':
                val = label_attributes[idx].get("VALIDATION")

                try:
                    val = eval(val)
                except NameError:
                    pass

                if val is True:
                    self._validation_ids.append(idx)
                elif val is False:
                    self._training_ids.append(idx)
                else:
                    Exception(
                        f"Sample in SET train must contain VALIDATION attribute. "
                        f"Validation must be True or False. "
                        f"Id: {idx}; VALIDATION={val}")

            elif split == 'test':
                self._testing_ids.append(idx)
            else:
                Exception(f"Labels FASTA header must contain SET. SET must be either 'train' or 'test'. "
                          f"Id: {idx}; SET={split}")

        # Infer classes from data
        class_labels = set()
        for classes in id2label.values():
            class_labels = class_labels | set(classes)

        self._class_labels = class_labels

        # Create a mapping from integers to class labels and reverse
        self._class_str2int = {letter: idx for idx, letter in enumerate(class_labels)}
        self._class_int2str = {idx: letter for idx, letter in enumerate(class_labels)}

        # Convert label values to lists of numbers based on the maps
        id2label = {identifier: np.array([self._class_str2int[label] for label in labels])
                    for identifier, labels in id2label.items()}  # classes idxs (zero-based)

        # Make sure the length of the sequence in the FASTA matches the length of the labels
        for seq_id, seq in id2fasta.items():
            if len(seq) != id2label[seq_id].size:
                Exception(f"Length mismatch for {seq_id}: Seq={len(seq)} VS Labels={id2label[seq_id].size}")

        self._id2fasta = id2fasta
        self._id2label = id2label

    def _retrieve_embeddings(
            self,
            embedder_name: Union[None, str] = "prottrans_bert_bfd",
            embeddings_file_path: Union[None, str] = None):

        # If embeddings don't exist, create them using the bio_embeddings pipeline
        if not Path(embeddings_file_path).is_file():
            out_config = execute_pipeline_from_config({
                "global": {
                    "sequence_file": self._sequence_file,
                    "prefix": "bio_embeddings_run",
                    "simple_remapping": True
                },
                "embeddings": {
                    "type": "embed",
                    "protocol": embedder_name
                }
            })

            embeddings_file_path = out_config['embeddings']['embeddings_file']

        # load pre-computed embeddings in .h5 file format computed via bio_embeddings
        logger.info(f"Loading embeddings from: {embeddings_file_path}")
        start = time.time()
        # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
        embeddings_file = h5py.File(embeddings_file_path, 'r', rdcc_nbytes=1024 ** 2 * 4000, rdcc_nslots=1e7)
        self._id2emb = {embeddings_file[idx].attrs["original_id"]: embedding for (idx, embedding) in embeddings_file.items()}

        # Logging
        logger.info(f"Read {len(self._id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.time() - start):.1f}[s]")

    def get_class_weights(self):
        # concatenate all labels irrespective of protein to count class sizes
        counter = Counter(list(itertools.chain.from_iterable(
            [list(labels) for labels in self._id2label.values()]
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

        return torch.FloatTensor(class_weights).to(self._device)

    def get_training_embeddings(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {idx: (torch.tensor(self._id2emb[idx]), torch.tensor(self._id2label[idx])) for idx in self._training_ids}

    def get_validation_embeddings(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            idx: (torch.tensor(self._id2emb[idx]), torch.tensor(self._id2label[idx])) for idx in self._validation_ids
        }

    def get_testing_embeddings(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {idx: (torch.tensor(self._id2emb[idx]), torch.tensor(self._id2label[idx])) for idx in self._testing_ids}

    def get_number_of_classes(self) -> int:
        return len(self._class_labels)

    def get_number_of_features(self) -> int:
        return torch.tensor(self._id2emb[self._training_ids[0]]).shape[1]
