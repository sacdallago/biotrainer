import torch
import numpy as np

from ..utilities import read_FASTA, get_sets_from_single_fasta

from .Trainer import Trainer


class SequenceTrainer(Trainer):

    @staticmethod
    def pipeline(**kwargs):
        return SequenceTrainer()._execute_pipeline(**kwargs)

    def _load_sequences_and_labels(self, sequence_file, labels_file):
        # Parse FASTA protein sequences
        protein_sequences = read_FASTA(sequence_file)
        id2fasta = {protein.id: str(protein.seq) for protein in protein_sequences}
        # Get the sets of labels, training, validation and testing samples
        id2label, training_ids, validation_ids, testing_ids = get_sets_from_single_fasta(protein_sequences)

        return training_ids, validation_ids, testing_ids, id2label, id2fasta

    def _generate_class_labels(self, id2label, id2fasta):
        # Infer classes from data
        class_labels = set(id2label.values())
        # Create a mapping from integers to class labels and reverse
        class_str2int = {letter: idx for idx, letter in enumerate(class_labels)}
        class_int2str = {idx: letter for idx, letter in enumerate(class_labels)}
        # Convert label values to lists of numbers based on the maps
        id2label = {identifier: np.array(class_str2int[label])
                    for identifier, label in id2label.items()}  # classes idxs (zero-based)

        return class_labels, id2label, class_int2str, class_str2int

    def _get_embeddings_config_and_file_name(self, sequence_file, output_dir, embedder_name):
        embeddings_config = {
            "global": {
                "sequences_file": sequence_file,
                "prefix": str(output_dir / embedder_name),
                "simple_remapping": True
            },
            "embeddings": {
                "type": "embed",
                "protocol": embedder_name,
                "reduce": True,
                "discard_per_amino_acid_embeddings": True
            }
        }
        embeddings_file_name = "reduced_embeddings_file.h5"

        return embeddings_config, embeddings_file_name

    def _get_number_features(self, id2emb, training_ids):
        return torch.tensor(id2emb[training_ids[0]]).shape[0]

    def _get_collate_function(self):
        return None
