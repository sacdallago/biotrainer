import torch
import numpy as np

from ..utilities import read_FASTA, get_sets_from_single_fasta

from .Trainer import Trainer


class SequenceTrainer(Trainer):

    @staticmethod
    def pipeline(**kwargs):
        return SequenceTrainer(**kwargs)._execute_pipeline()

    def _load_sequences_and_labels(self):
        # Parse FASTA protein sequences
        protein_sequences = read_FASTA(self.sequence_file)
        self.id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}
        # Get the sets of labels, training, validation and testing samples
        self.id2label, \
            self.training_ids, self.validation_ids, self.testing_ids = get_sets_from_single_fasta(protein_sequences)

    def _generate_class_labels(self):
        # Infer classes from data
        self.class_labels = set(self.id2label.values())
        # Create a mapping from integers to class labels and reverse
        self.class_str2int = {letter: idx for idx, letter in enumerate(self.class_labels)}
        self.class_int2str = {idx: letter for idx, letter in enumerate(self.class_labels)}
        # Convert label values to lists of numbers based on the maps
        self.id2label = {identifier: np.array(self.class_str2int[label])
                         for identifier, label in self.id2label.items()}  # classes idxs (zero-based)

    def _use_reduced_embeddings(self) -> bool:
        return True

    def _get_number_features(self) -> int:
        return torch.tensor(self.id2emb[self.training_ids[0]]).shape[0]

    def _get_collate_function(self):
        return None
