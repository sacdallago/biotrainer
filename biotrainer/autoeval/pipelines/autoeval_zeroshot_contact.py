from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from biotrainer_core.data_classes import ZeroShotMethod, ContactSingleProteinResult
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalProgress, \
    ContactFrameworkReport, ZeroShotContactCachedResults

from biotrainer_core.input_files import read_FASTA, load_contact_map

from .autoeval_pipeline_utils import subsample_seq_records_for_contact_development_mode

from ..core import AutoEvalFramework

from ...bioengineer import BioEngineer
from ...shared.metrics import evaluate_contact_dataset


def autoeval_zeroshot_contact_pipeline(framework: AutoEvalFramework,
                                       embedder_name: str,
                                       zero_shot_method: ZeroShotMethod,
                                       output_dir: Path,
                                       autoeval_tasks: List[Tuple[AutoEvalTask, Dict[str, Any]]],
                                       bioengineer: Optional[BioEngineer],
                                       development_mode: bool,
                                       device=None,
                                       ):
    assert bioengineer is not None, f"BioEngineer could not be initialized for embedder {embedder_name}!"

    # Load cached results 
    zero_shot_contact_cached_results = ZeroShotContactCachedResults.loaded_or_empty(embedder_name=embedder_name,
                                                                                    method=zero_shot_method,
                                                                                    output_dir=output_dir)
    # Execute bioengineer
    autoeval_tasks = [task for task, _ in autoeval_tasks]  # Ignore config for zeroshot contact
    assert len(autoeval_tasks) == 1, "Only one supervised contact task is supported!"
    task = autoeval_tasks[0]

    task_names = [task.combined_name() for task in autoeval_tasks]
    print(f"The following tasks will be executed in order: {task_names} (total {len(task_names)})")
    total_tasks = len(task_names)
    current_task_name = ""
    current_task_name = task.combined_name()
    print(f"Running task {current_task_name}...")
    yield AutoEvalProgress(completed_tasks=0,
                           total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name())

    assert len(task.input_files) == 1, (f"Empty/Multiple input files/dirs not "
                                        f"supported for contact tasks! Task: {current_task_name}")

    dataset_dir_path = task.input_files[0]
    fasta_file_path = dataset_dir_path / "extracted_sequences.fasta"
    contacts_dir_path = dataset_dir_path / "contacts"

    # Read fasta file
    seq_records = read_FASTA(fasta_file_path)
    if len(seq_records) == 0:
        raise ValueError(f"Fasta file {fasta_file_path} is empty!")

    seq_records_subsampled = subsample_seq_records_for_contact_development_mode(seq_records)
    development_ids = [seq_record.seq_id for seq_record in seq_records_subsampled]
    if development_mode:
        seq_records = seq_records_subsampled
    zero_shot_contact_framework_report = ContactFrameworkReport.empty(method=zero_shot_method,
                                                                      development_ids=development_ids)

    # Check for cached results
    cached_results: Dict[str, ContactSingleProteinResult] = {}
    for seq_record in seq_records:
        seq_id = seq_record.seq_id
        # Check if cached result exists for this dataset
        maybe_cached_result = zero_shot_contact_cached_results.maybe_cached_result(seq_id=seq_id)
        if maybe_cached_result is not None:
            print(f"Skipping dataset {seq_id} as cached result exists for {seq_id}!")
            cached_results[seq_id] = maybe_cached_result

    # Define loading of ground truth contact map
    def load_gt_contact_map(seq_record):
        seq = seq_record.seq
        seq_id = seq_record.seq_id
        ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
        return load_contact_map(path=ground_truth_contact_map_path,
                                sequence=seq,
                                structure_id=seq_id)

    # Define evaluation function
    def evaluate():
        yield from evaluate_contact_dataset(dataset_name=current_task_name,
                                            items=seq_records,
                                            predict_func=lambda
                                                seq_record: bioengineer.zero_shot_contact_map(
                                                method=zero_shot_method,
                                                sequence=seq_record.seq),
                                            get_ground_truth_func=load_gt_contact_map,
                                            get_seq_id_func=lambda d: d.seq_id,
                                            cached_results=cached_results,
                                            )

    # Run Evaluation of zero-shot contact maps
    for maybe_single_result, maybe_dataset_result in evaluate():
        if maybe_single_result is not None:
            zero_shot_contact_cached_results.update_and_sync(result=maybe_single_result,
                                                             output_dir=output_dir)
        if maybe_dataset_result is not None:
            zero_shot_contact_framework_report.update_result(task_name=current_task_name,
                                                             dataset_result=maybe_dataset_result)
            break  # Done

    print(f"Finished task {current_task_name}!")

    print(f"Autoeval zeroshot contact pipeline on framework {framework.get_name()} "
          f"for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks,
                           total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=zero_shot_contact_framework_report)
