import torch
import random

from typing import List, Union, Dict

from ..utilities import EmbeddingDatasetSample, SequenceDatasetSample, MASK_AND_LABELS_PAD_VALUE


class BiotrainerDataset(torch.utils.data.Dataset):
    ids: List[str]
    inputs: Union[List[torch.tensor], List[str]]  # Embeddings / Sequences
    targets: List[torch.tensor]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        seq_id = self.ids[index]
        x = self.inputs[index]
        y = self.targets[index]
        return seq_id, x, y


class EmbeddingsDataset(BiotrainerDataset):
    """Embeddings dataset for training on pre-calculated embeddings."""
    def __init__(self, samples: List[EmbeddingDatasetSample]):
        self.ids, self.inputs, self.targets = zip(
            *[(sample.seq_id, sample.embedding, sample.target) for sample in samples]
        )


class SequenceDataset(BiotrainerDataset):
    """Sequence dataset for fine-tuning"""
    def __init__(self, samples: List[SequenceDatasetSample],):
        self.ids, self.inputs, self.targets = zip(
            *[(sample.seq_id, sample.seq_record.seq, sample.target) for sample in samples]
        )


class SequenceDatasetWithRandomMasking(BiotrainerDataset):
    """Sequence dataset for fine-tuning with random masking"""
    def __init__(self, samples: List[SequenceDatasetSample], mask_token: str, class_str2int: Dict[str, int]):
        self.ids, self.inputs, self.targets = zip(
            *[(sample.seq_id, sample.seq_record.seq, sample.target) for sample in samples]
        )
        self.mask_token = mask_token
        self.class_str2int = class_str2int  # To convert amino acid to class integer

    def __getitem__(self, item):
        seq_id = self.ids[item]
        x = self.inputs[item]

        masked_seq, masked_target = self.random_masking(x, self.mask_token, self.class_str2int)

        return seq_id, masked_seq, masked_target

    @staticmethod
    def random_masking(sequence: str, mask_token: str, class_str2int: Dict[str, int]):
        """
        Apply BERT-style random masking to a sequence: https://arxiv.org/pdf/1810.04805.
        Returns masked sequence and corresponding labels for loss calculation.
        """
        masking_probability = 0.15
        nested_probability_mask = 0.8
        nested_probability_random_aa = 0.1
        nested_probability_id = 0.1

        assert (nested_probability_mask + nested_probability_random_aa + nested_probability_id == 1)

        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        # Convert to lists for manipulation
        sequence_list = list(sequence)
        masked_sequence = sequence_list.copy()

        # Create target labels - initialize with MASK_AND_LABELS_PAD_VALUE
        masked_labels = [MASK_AND_LABELS_PAD_VALUE] * len(sequence_list)

        def _get_replacement_token(index: int):
            replacement_probability = random.random()
            if replacement_probability < nested_probability_mask:
                return mask_token  # Replace with mask
            elif replacement_probability < nested_probability_mask + nested_probability_random_aa:
                return random.choice(amino_acids)  # Replace with random aa
            else:
                return sequence_list[index] # Keep original token (chance: nested_probability_id)


        # Apply masking
        for i in range(len(sequence_list)):
            do_masking = random.random() < masking_probability
            if do_masking:
                # Store original token as target for this position
                masked_labels[i] = class_str2int[sequence_list[i]]

                # Replace aa
                sequence_list[i] = _get_replacement_token(i)

        # Ensure at least one position is masked
        if all(label == MASK_AND_LABELS_PAD_VALUE for label in masked_labels):
            random_pos = random.randint(0, len(sequence_list) - 1)
            masked_labels[random_pos] = class_str2int[sequence_list[random_pos]]
            masked_sequence[random_pos] = _get_replacement_token(random_pos)

        # Convert back to string
        masked_sequence = ''.join(masked_sequence)

        return masked_sequence, torch.tensor(masked_labels)