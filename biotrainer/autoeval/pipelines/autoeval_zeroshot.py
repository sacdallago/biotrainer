from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from biotrainer_core.data_classes import ZeroShotMethod
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalProgress, \
    ZeroShotFrameworkReport, ZeroShotCachedResults

from ..core import AutoEvalFramework

from ...shared import get_device
from ...bioengineer import BioEngineer


def autoeval_zeroshot_pipeline(framework: AutoEvalFramework,
                               embedder_name: str,
                               zero_shot_method: ZeroShotMethod,
                               output_dir: Path,
                               autoeval_tasks: List[Tuple[AutoEvalTask, Dict[str, Any]]],
                               bioengineer: Optional[BioEngineer] = None,
                               device=None):
    if not bioengineer:
        bioengineer = BioEngineer.from_name(name=embedder_name, device=get_device(device))

    # Load cached results
    cached_results = ZeroShotCachedResults.loaded_or_empty(embedder_name=embedder_name,
                                                           method=zero_shot_method,
                                                           output_dir=output_dir)
    # Execute bioengineer
    development_ids = framework.get_data_handler()._file_paths_dev_mode  # TODO: Not an ideal solution to access dev ids like this
    assert len(development_ids) > 0, "Development ids must be provided"

    zero_shot_framework_report = ZeroShotFrameworkReport.empty(method=zero_shot_method,
                                                               development_ids=development_ids)
    autoeval_tasks = [task for task, _ in autoeval_tasks]  # Ignore config for zero shot
    task_names = [task.combined_name() for task in autoeval_tasks]
    print(f"The following tasks will be executed in order: {task_names} (total {len(task_names)})")
    completed_tasks = 0
    total_tasks = len(task_names)
    current_task_name = ""
    for task in autoeval_tasks:
        current_task_name = task.combined_name()
        print(f"Running task {current_task_name}...")
        yield AutoEvalProgress(completed_tasks=completed_tasks,
                               total_tasks=total_tasks,
                               current_task_name=current_task_name,
                               current_framework_name=framework.get_name())
        individual_results = {}
        for idx, file_path in enumerate(task.input_files):
            print(f"Running dataset {idx + 1}/{len(task.input_files)} [name: {file_path.name}]...")
            # Check if cached result exists for this dataset
            file_name = file_path.name
            maybe_cached_result = cached_results.maybe_cached_result(dataset_name=file_name)
            if maybe_cached_result is not None:
                print(f"Skipping dataset {file_name} as cached result exists for {file_name}!")
                individual_results[file_name] = maybe_cached_result
                continue

            # Cached result does not exist, run bioengineer
            _, ranking_result = bioengineer.rank_pgym_dataset(dataset_file_path=file_path,
                                                              method=zero_shot_method,
                                                              single_mutations_only=False)
            cached_results.update_and_sync(dataset_name=file_name, result=ranking_result, output_dir=output_dir)
            individual_results[file_name] = ranking_result

        # Aggregate results
        zero_shot_framework_report.aggregate(task_name=current_task_name, individual_results=individual_results)
        completed_tasks += 1
        print(f"Finished task {current_task_name}!")

    print(f"Autoeval zeroshot pipeline on framework {framework.get_name()} "
          f"for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=zero_shot_framework_report)
