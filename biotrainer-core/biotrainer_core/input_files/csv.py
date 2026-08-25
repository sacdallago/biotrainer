import pandas as pd

from pathlib import Path
from typing import Union, Optional, List

from .fasta import write_FASTA

from ..data_classes import SequenceData, Variant, VariantScore


def pgym_csv_to_fasta(pgym_csv_path: Union[str, Path],
                      output_fasta_path: Union[str, Path],
                      single_mutations_only: Optional[bool] = False,
                      use_binary_score: Optional[bool] = False,
                      ) -> int:
    # Read ProteinGym dataset
    df = pd.read_csv(pgym_csv_path)
    if len(df) == 0:
        raise ValueError(f"Dataset file {pgym_csv_path} is empty!")

    try:
        score_column = "DMS_score_bin" if use_binary_score else "DMS_score"

        mutation_fitness = {row["mutated_sequence"]: (row["mutant"], row[score_column]) for _, row in df.iterrows()}
        if single_mutations_only:
            mutation_fitness = {seq: (mut, score) for seq, (mut, score) in mutation_fitness.items() if ":" not in mut}
    except KeyError as e:
        raise ValueError(f"Dataset file {pgym_csv_path} is missing a required column: {e}")

    sequence_data = [SequenceData(seq_id=f"mut_{mut}", seq=seq, label=str(score))
                     for seq, (mut, score) in mutation_fitness.items()]

    return write_FASTA(output_fasta_path, sequence_data)


def parse_pgym_file(pgym_csv_path: Union[str, Path],
                    single_mutations_only: bool = False) -> List[VariantScore]:
    # Read ProteinGym dataset
    df = pd.read_csv(pgym_csv_path)
    if len(df) == 0:
        raise ValueError(f"Dataset file {pgym_csv_path} is empty!")

    try:
        first_row = df.iloc[0]
        mt_seq = first_row["mutated_sequence"]
        mutation_string = first_row["mutant"]
        mutation_fitness = {row["mutant"]: row["DMS_score"] for _, row in df.iterrows()}
        if single_mutations_only:
            mutation_fitness = {mut: score for mut, score in mutation_fitness.items() if ":" not in mut}
    except KeyError as e:
        raise ValueError(f"Dataset file {pgym_csv_path} is missing a required column: {e}")

    # Derive wild-type sequence
    one_indexed = True  # ProteinGym default
    wt_seq = Variant.derive_wildtype_sequence(mutation_sequence=mt_seq, variant_string=mutation_string,
                                              one_indexed=one_indexed)
    print(f"Wild-type sequence for {pgym_csv_path} was derived as {wt_seq}.")

    variant_scores = [VariantScore.from_experimental(variant=Variant.parse(variant_string=mutation),
                                                     experimental_score=score)
                      for mutation, score in mutation_fitness.items()]
    return variant_scores