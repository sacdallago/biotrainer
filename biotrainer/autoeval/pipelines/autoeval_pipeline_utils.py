import random

from typing import List, Set
from biotrainer_core.data_classes import SequenceData, ContactDatasetResult, ContactSingleProteinResult

def get_dataset_and_dev_result_from_single_contact_results(dataset_name: str,
                                                           single_results: List[ContactSingleProteinResult],
                                                           ):
    bootstrap_iterations = 10000  # High K because of cheap evaluation
    bootstrap_seed = 44
    confidence_level = 0.05
    dataset_result = ContactDatasetResult.aggregate(dataset_name=dataset_name,
                                                    per_protein_results=single_results,
                                                    iterations=bootstrap_iterations,
                                                    seed=bootstrap_seed,
                                                    confidence_level=confidence_level
                                                    )

    return dataset_result
