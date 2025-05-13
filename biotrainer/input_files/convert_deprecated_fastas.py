from typing import Optional

from .fasta import read_FASTA

def convert_deprecated_fastas(result_file: str,
                              sequence_file: str,
                              labels_file: Optional[str] = None,
                              masks_file: Optional[str] = None,
                              ):
    seqs = read_FASTA(sequence_file)
    labels = None
    masks = None

    if labels_file is not None:
        labels = read_FASTA(labels_file)
    if masks_file is not None:
        masks = read_FASTA(masks_file)

    result_string = ""
    for seq_id, seq_record in seqs.items():
        merged_target = None
        merged_set = None
        merged_mask = None
        # TARGET
        if labels and seq_id in labels:
            if seq_record.get_target() is not None:
                raise ValueError("Trying to merge a sequence file with TARGET=.. annotations with a labels file. "
                                 "This behaviour is not expected, please remove the TARGET "
                                 "annotations in the sequence file first!")
            merged_target = labels[seq_id].seq
        else:
            merged_target = seq_record.get_target()

        # SET
        if labels and seq_id in labels:
            seq_set = seq_record.get_set()
            labels_set = labels[seq_id].get_set()
            if seq_set != labels_set and not (seq_set is not None or labels_set is not None):
                raise ValueError(f"Found ambiguity between sets for {seq_id} in sequences (SET={seq_set}) "
                                 f"and labels (SET={labels_set})!")
            else:
                merged_set = labels_set if labels_set is not None else seq_set

        # MASKS
        if masks and seq_id in masks:
            if seq_record.get_mask() is not None:
                raise ValueError("Trying to merge a sequence file with MASK=.. annotations with a masks file. "
                                 "This behaviour is not expected, please remove the MASK "
                                 "annotations in the sequence file first!")
            merged_mask = masks[seq_id].seq

        if merged_target is None:
            raise ValueError(f"Could not merge target for seq_id {seq_id}!")
        if merged_set is None:
            raise ValueError(f"Could not merge set for seq_id {seq_id}!")

        merged_header = f">{seq_id} TARGET={merged_target} SET={merged_set}"
        if merged_mask:
            merged_header += f" MASK={merged_mask}"
        if seq_record.seq is None or seq_record.seq == "":
            raise ValueError(f"Could not find a valid sequence for seq_id {seq_id}!")
        result_string += merged_header + f"\n{seq_record.seq}\n"

    with open(result_file, "w") as f:
        f.write(result_string)
