import torch
import numpy as np

from ..utilities import read_FASTA, get_sets_from_labels
from ..datasets import pad_sequences

from .Trainer import Trainer


class ResidueTrainer(Trainer):

    @staticmethod
    def pipeline(**kwargs):
        return ResidueTrainer(**kwargs)._execute_pipeline()

    def _load_sequences_and_labels(self):
        # Parse FASTA protein sequences
        protein_sequences = read_FASTA(self.sequence_file)
        self.id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}

        # Parse label sequences
        label_sequences = read_FASTA(self.labels_file)
        self.id2label = {label.id: str(label.seq) for label in label_sequences}
        # Get the sets of training, validation and testing samples
        self.training_ids, self.validation_ids, self.testing_ids = get_sets_from_labels(label_sequences)

    def _generate_class_labels(self):
        # Infer classes from data
        self.class_labels = set()
        for classes in self.id2label.values():
            self.class_labels = self.class_labels | set(classes)

        # Create a mapping from integers to class labels and reverse
        self.class_str2int = {letter: idx for idx, letter in enumerate(self.class_labels)}
        self.class_int2str = {idx: letter for idx, letter in enumerate(self.class_labels)}

        # Convert label values to lists of numbers based on the maps
        self.id2label = {identifier: np.array([self.class_str2int[label] for label in labels])
                         for identifier, labels in self.id2label.items()}  # classes idxs (zero-based)

        # Make sure the length of the sequences in the FASTA matches the length of the sequences in the labels
        for seq_id, seq in self.id2fasta.items():
            if len(seq) != self.id2label[seq_id].size:
                Exception(f"Length mismatch for {seq_id}: Seq={len(seq)} VS Labels={self.id2label[seq_id].size}")

    def _use_reduced_embeddings(self) -> bool:
        return False

    def _get_number_features(self):
        return torch.tensor(self.id2emb[self.training_ids[0]]).shape[1]

    def _get_collate_function(self):
        return pad_sequences
