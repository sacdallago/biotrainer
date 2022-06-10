import logging
import numpy as np

from typing import Dict, Any, Optional, List, Set
from Bio.SeqRecord import SeqRecord

from ..utilities import read_FASTA, get_attributes_from_seqrecords

logger = logging.getLogger(__name__)


class TargetManager:
    id2target: Dict[str, Any] = dict()
    id2attributes: Dict[str, Any] = dict()
    number_of_outputs: int = 1

    # Optional
    class_str2int: Optional[Dict[str, int]] = None
    class_int2str: Optional[Dict[int, str]] = None
    class_labels: Optional[Set[str]] = None

    def __init__(self, protocol: str, protein_sequences: List[SeqRecord], labels_file: Optional[str] = None):

        # 1. Residue Level
        if 'residue_' in protocol:
            # Expect labels file to be in FASTA format, with each "residue" being the residue-associated-label
            label_sequences = read_FASTA(labels_file)
            self.id2attributes = get_attributes_from_seqrecords(label_sequences)

            # Generate Mapping from Ids to Labels
            self.id2target = {label.id: str(label.seq) for label in label_sequences}

            # a) Class output
            if 'class' in protocol:
                self.class_labels = set()

                # Infer classes from data
                for classes in self.id2target.values():
                    self.class_labels = self.class_labels | set(classes)

                self.number_of_outputs = len(self.class_labels)

                # Create a mapping from integers to class labels and reverse
                self.class_str2int = {letter: idx for idx, letter in enumerate(self.class_labels)}
                self.class_int2str = {idx: letter for idx, letter in enumerate(self.class_labels)}

                # Convert label values to lists of numbers based on the maps
                self.id2target = {identifier: np.array([self.class_str2int[label] for label in labels])
                                  for identifier, labels in self.id2target.items()}  # classes idxs (zero-based)

            # b) Value output
            else:
                raise NotImplementedError

            # Make sure the length of the sequences in the FASTA matches the length of the sequences in the labels
            for sequence in protein_sequences:
                if len(sequence) != self.id2target[sequence.id].size:
                    Exception(
                        f"Length mismatch for {sequence.id}: "
                        f"Seq={len(sequence)} VS Labels={self.id2target[sequence.id].size}"
                    )

        # 2. Sequence Level
        elif 'sequence_' in protocol:

            # In sequence task, split definitions are in sequence header, as well as target
            # For more info check file specifications!
            self.id2attributes = get_attributes_from_seqrecords(protein_sequences)

            self.id2target = {seq_id: seq_vals["TARGET"] for seq_id, seq_vals in self.id2attributes.items()}
            # a) Class output
            if 'class' in protocol:
                # Infer classes from data
                self.class_labels = set(self.id2target.values())
                self.number_of_outputs = len(self.class_labels)

                # Create a mapping from integers to class labels and reverse
                self.class_str2int = {letter: idx for idx, letter in enumerate(self.class_labels)}
                self.class_int2str = {idx: letter for idx, letter in enumerate(self.class_labels)}

                # Convert label values to lists of numbers based on the maps
                self.id2target = {identifier: np.array(self.class_str2int[label])
                                  for identifier, label in self.id2target.items()}  # classes idxs (zero-based)

            # b) Value output
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        if not self.id2target:
            raise Exception("Prediction targets not found or could not be extracted!")
