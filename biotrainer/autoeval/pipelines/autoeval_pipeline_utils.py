import random

from typing import List, Set
from biotrainer_core.data_classes import SequenceData, ContactDatasetResult, ContactSingleProteinResult


def subsample_seq_records_for_contact_development_mode(seq_records: List[SequenceData]) -> List[SequenceData]:
    initial_n = len(seq_records)
    sample_ratio_casp = 0.4  # Higher ratio for the rather small datasets to keep a meaningful sample
    sample_ratio_selected = 0.05
    if initial_n < 100:
        sample_ratio = sample_ratio_casp
    else:
        sample_ratio = sample_ratio_selected
    rng = random.Random(14)
    sample = rng.sample(seq_records, int(len(seq_records) * sample_ratio))

    print(f"Subsampled {initial_n} sequences to {len(sample)} for contact development mode.")
    return sample

def get_dataset_and_dev_result_from_single_contact_results(dataset_name: str,
                                                           single_results: List[ContactSingleProteinResult],
                                                           development_ids: Set[str]):
    bootstrap_iterations = 10000  # High K because of cheap evaluation
    bootstrap_seed = 44
    confidence_level = 0.05
    dataset_result = ContactDatasetResult.aggregate(dataset_name=dataset_name,
                                                    per_protein_results=single_results,
                                                    iterations=bootstrap_iterations,
                                                    seed=bootstrap_seed,
                                                    confidence_level=confidence_level
                                                    )
    single_results_dev_only = [single_result for single_result in single_results
                               if single_result.protein_name in development_ids]
    dataset_result_dev = ContactDatasetResult.aggregate(dataset_name=dataset_name,
                                                        per_protein_results=single_results_dev_only,
                                                        iterations=bootstrap_iterations,
                                                        seed=bootstrap_seed,
                                                        confidence_level=confidence_level
                                                        )

    return dataset_result, dataset_result_dev