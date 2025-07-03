from typing import Optional

from .fasta import read_FASTA

def convert_deprecated_fastas(result_file: str,
                              sequence_file: str,
                              labels_file: Optional[str] = None,
                              masks_file: Optional[str] = None,
                              skip_sequence_on_failed_merge: bool = False,
                              ):
    seqs = read_FASTA(sequence_file)
    labels = None
    masks = None

    if labels_file is not None:
        labels = {label_record.seq_id: label_record for label_record in read_FASTA(labels_file)}
    if masks_file is not None:
        masks = {mask_record.seq_id: mask_record for mask_record in read_FASTA(masks_file)}

    result_string = ""
    for seq_record in seqs:
        seq_id = seq_record.seq_id
        merged_target = seq_record.get_target()
        merged_set = seq_record.get_deprecated_set()
        merged_mask = seq_record.get_mask()
        # TARGET
        if labels and seq_id in labels:
            label_target = labels[seq_id].seq
            if merged_target is not None and merged_target != label_target:
                print(f"OVERWRITING sequence file TARGET for sequence {seq_id}!")
            merged_target = label_target

        # SET
        if labels and seq_id in labels:
            seq_set = seq_record.get_deprecated_set()
            labels_set = labels[seq_id].get_deprecated_set()
            if seq_set != labels_set and not (seq_set is not None or labels_set is not None):
                raise ValueError(f"Found ambiguity between sets for {seq_id} in sequences (SET={seq_set}) "
                                 f"and labels (SET={labels_set})!")
            else:
                merged_set = labels_set if labels_set is not None else seq_set

        # MASKS
        if masks and seq_id in masks:
            mask_file_mask = masks[seq_id].seq
            if merged_mask is not None and merged_mask != mask_file_mask:
                print(f"OVERWRITING mask file mask for sequence {seq_id}!")
            merged_mask = mask_file_mask

        # VALIDATE
        if merged_target is None:
            if skip_sequence_on_failed_merge:
                print(f"Could not merge target for seq_id {seq_id}, skipping sequence!")
                continue
            raise ValueError(f"Could not merge target for seq_id {seq_id}!")

        if labels_file and len(merged_target) != len(seq_record.seq):
            if skip_sequence_on_failed_merge:
                print(f"Sequence {seq_id} has incorrect target length, skipping sequence!")
                continue
            raise ValueError(f"Sequence {seq_id} has incorrect target length: "
                             f"{len(merged_target)} != {len(seq_record.seq)}!")

        if masks_file and merged_mask is not None and (len(merged_mask) != len(merged_target)):
            if skip_sequence_on_failed_merge:
                print(f"Sequence {seq_id} has incorrect mask length, skipping sequence!")
                continue
            raise ValueError(f"Sequence {seq_id} has incorrect mask length: "
                             f"{len(merged_mask)} != {len(merged_target)}!")

        if merged_set is None:
            if skip_sequence_on_failed_merge:
                print(f"Could not merge set for seq_id {seq_id}, skipping sequence!")
                continue
            raise ValueError(f"Could not merge set for seq_id {seq_id}!")

        merged_header = f">{seq_id} TARGET={merged_target} SET={merged_set}"
        if merged_mask:
            merged_header += f" MASK={merged_mask}"
        if seq_record.seq is None or seq_record.seq == "":
            raise ValueError(f"Could not find a valid sequence for seq_id {seq_id}!")
        result_string += merged_header + f"\n{seq_record.seq}\n"

    with open(result_file, "w") as f:
        f.write(result_string)
