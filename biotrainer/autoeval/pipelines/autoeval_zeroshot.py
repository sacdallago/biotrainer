from pathlib import Path
from typing import Optional, Union, List

from .autoeval_setup import setup_pipeline
from .autoeval_report import ZeroShotCachedResults, ZeroShotFrameworkReport, AutoEvalReport
from .autoeval_progress import AutoEvalProgress
from ..core import AutoEvalFramework, AutoEvalTask

from ...utilities import get_device
from ...bioengineer import BioEngineer, ZeroShotMethod


def _run_tasks(framework: AutoEvalFramework,
               embedder_name: str,
               zero_shot_method: ZeroShotMethod,
               autoeval_report: AutoEvalReport,
               output_dir: Path,
               autoeval_tasks: List[AutoEvalTask],
               bioengineer: Optional[BioEngineer] = None,
               device=None):
    if not bioengineer:
        bioengineer = BioEngineer.from_name(name=embedder_name, device=get_device(device))

    # Load cached results
    cached_results = ZeroShotCachedResults.loaded_or_empty(embedder_name=embedder_name,
                                                           method=zero_shot_method,
                                                           output_dir=output_dir)
    # Execute bioengineer
    zero_shot_framework_report = ZeroShotFrameworkReport.empty(method=zero_shot_method)
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
            ranking_result = bioengineer.rank_pgym_dataset(dataset_file_path=file_path,
                                                           method=zero_shot_method,
                                                           single_mutations_only=False)
            cached_results.update_and_sync(dataset_name=file_name, result=ranking_result, output_dir=output_dir)
            individual_results[file_name] = ranking_result

        # Aggregate results
        zero_shot_framework_report.aggregate(task_name=current_task_name, individual_results=individual_results)
        completed_tasks += 1
        print(f"Finished task {current_task_name}!")

    autoeval_report.add_zeroshot_result(framework_name=framework.get_name(), report=zero_shot_framework_report)
    autoeval_report.write(output_dir=output_dir.parent)

    print(f"Autoeval pipeline on framework {framework.get_name()} for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=autoeval_report)


def autoeval_zeroshot_pipeline(embedder_name: str,
                               framework: AutoEvalFramework,
                               method: ZeroShotMethod,
                               autoeval_report: AutoEvalReport,
                               output_dir: Optional[Union[Path, str]] = "autoeval_output",
                               force_download: Optional[bool] = False,
                               custom_storage_path: Optional[Union[Path, str]] = None,
                               custom_bioengineer: Optional[BioEngineer] = None,
                               device=None,
                               ):
    # Setup
    autoeval_tasks = setup_pipeline(data_handler=framework.get_data_handler(),
                                    custom_storage_path=custom_storage_path,
                                    force_download=force_download)
    # Pipeline
    yield from _run_tasks(framework=framework,
                          embedder_name=embedder_name,
                          zero_shot_method=method,
                          autoeval_report=autoeval_report,
                          output_dir=output_dir,
                          autoeval_tasks=autoeval_tasks,
                          bioengineer=custom_bioengineer,
                          device=device)
