import os

from pathlib import Path
from biotrainer_core.input_files import read_FASTA
from typing import Callable, List, Optional, Union, Any, Dict, Tuple
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalMode
from biotrainer_core.data_classes import ZeroShotMethod, SequenceData, Protocol

from ..core import AutoEvalDataHandler, AutoEvalFramework
from ..autoeval_frameworks import framework_factory, AvailableFramework


def setup_output_dir(base_dir: Path, embedder_name: str, framework_name: str) -> Path:
    embedder_dir_name = embedder_name
    if "/" in embedder_dir_name:  # Huggingface
        embedder_dir_name = embedder_dir_name.replace("/", "-")

    output_dir = Path(base_dir) / embedder_dir_name / framework_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def validate_input(framework,
                   zero_shot_method: Optional[ZeroShotMethod],
                   min_seq_length: Optional[int],
                   max_seq_length: Optional[int]) -> AutoEvalFramework:
    framework_obj = framework_factory(framework)

    if framework_obj is None:
        raise ValueError(f"Unsupported framework: {framework}")

    match framework_obj.get_mode():
        case AutoEvalMode.SUPERVISED:  # Supervised frameworks
            if zero_shot_method is not None:
                raise ValueError("Zero-shot method must not be provided for a supervised framework!")
            if min_seq_length is None or max_seq_length is None:
                raise ValueError("min_seq_length and max_seq_length must be provided for a supervised framework!")
            if min_seq_length >= max_seq_length:
                raise ValueError("min_seq_length must be less than max_seq_length")
            if max_seq_length <= 0:
                raise ValueError("max_seq_length must be greater than 0")
        case AutoEvalMode.ZERO_SHOT:  # Zero-Shot frameworks
            if zero_shot_method is None:
                raise ValueError("Zero-shot method must be provided for a zero-shot framework!")
            if zero_shot_method == ZeroShotMethod.JACOBIAN_CONTACT:
                raise ValueError(
                    "Zero-shot method JACOBIAN_CONTACT currently only supported in mode ZERO_SHOT_CONTACT!")
        case AutoEvalMode.ZERO_SHOT_CONTACT:  # Zero-Shot contact frameworks
            if zero_shot_method is None:
                raise ValueError("Zero-shot method must be provided for a zero-shot framework!")
            if zero_shot_method != ZeroShotMethod.JACOBIAN_CONTACT:
                raise ValueError(
                    "Only zero-shot method JACOBIAN_CONTACT currently supported in mode ZERO_SHOT_CONTACT!")

    return framework_obj


def _apply_task_filter(tasks: List[AutoEvalTask],
                       task_filter: Callable[[AutoEvalTask], bool],
                       framework_name: str) -> List[AutoEvalTask]:
    """ Keep only the tasks the filter selects, refusing to run nothing at all.

    A filter that matches no task would otherwise produce an empty but successful run, which then gets written
    as a framework report and reused - so fail here, naming what could have been selected.
    """
    selected = [task for task in tasks if task_filter(task)]
    if not selected:
        raise ValueError(f"task_filter selected none of the {len(tasks)} tasks of framework {framework_name}. "
                         f"Available: {[task.combined_name() for task in tasks]}")
    return selected


def get_unique_framework_sequences(framework: Union[str, AvailableFramework, AutoEvalFramework],
                                   min_seq_length: int,
                                   max_seq_length: int,
                                   custom_storage_path: Optional[Union[Path, str]] = None,
                                   force_download: Optional[bool] = False,
                                   development_mode: bool = True,
                                   task_filter: Optional[Callable[[AutoEvalTask], bool]] = None,
                                   ) -> Tuple[
    List[Tuple[AutoEvalTask, Dict[str, Any]]], Dict[str, SequenceData],
    Dict[str, SequenceData]]:
    framework_obj = framework
    if not isinstance(framework_obj, AutoEvalFramework):
        framework_obj = validate_input(framework,
                                       zero_shot_method=None,
                                       min_seq_length=min_seq_length,
                                       max_seq_length=max_seq_length)

    data_handler = framework_obj.get_data_handler()
    config_bank = framework_obj.get_config_bank()
    auto_eval_tasks = setup_pipeline(data_handler=data_handler,
                                     min_seq_length=min_seq_length,
                                     max_seq_length=max_seq_length,
                                     custom_storage_path=custom_storage_path,
                                     force_download=force_download,
                                     development_mode=development_mode,)
    if task_filter:
        # Filter before the configs and the unique sequences are collected, so that pre-embedding shrinks with
        # the selection instead of covering the whole framework
        auto_eval_tasks = _apply_task_filter(tasks=auto_eval_tasks,
                                             task_filter=task_filter,
                                             framework_name=framework_obj.get_name())
    task_config_tuples = []
    for task in auto_eval_tasks:
        config = config_bank.get_task_config(task=task)
        task_config_tuples.append((task, config))

    # Get unique sequences for supervised tasks for pre-embedding
    unique_per_residue = {}
    unique_per_sequence = {}
    if framework_obj.get_mode() == AutoEvalMode.SUPERVISED:
        unique_per_residue, unique_per_sequence = _get_unique_sequences_for_all_tasks(
            {str(t.input_files[0]): Protocol.from_string(c["protocol"]) for t, c in task_config_tuples}
        )
    return task_config_tuples, unique_per_residue, unique_per_sequence


def _get_unique_sequences_for_all_tasks(tasks: Dict[str, Protocol]) -> Tuple[
    Dict[str, SequenceData], Dict[str, SequenceData]]:
    unique_per_residue = {}
    unique_per_sequence = {}
    for task_input, protocol in tasks.items():
        seq_records = read_FASTA(task_input)
        for seq_record in seq_records:
            if protocol in Protocol.using_per_sequence_embeddings():
                unique_per_sequence[seq_record.get_hash()] = seq_record
            else:
                unique_per_residue[seq_record.get_hash()] = seq_record
    return unique_per_residue, unique_per_sequence


def setup_pipeline(data_handler: AutoEvalDataHandler,
                   min_seq_length: Optional[int] = None,
                   max_seq_length: Optional[int] = None,
                   custom_storage_path: Optional[Union[Path, str]] = None,
                   force_download: Optional[bool] = False,
                   development_mode: bool = True,
                   ) -> List[AutoEvalTask]:
    framework_base_path = data_handler.get_framework_base_path(
        custom_storage_path=custom_storage_path)

    if force_download:
        data_handler.clear_autoeval_cache()

    if not os.path.exists(framework_base_path):
        os.makedirs(framework_base_path, exist_ok=True)

    if data_handler.is_download_necessary(framework_base_path):
        data_handler.download_data(data_dir=framework_base_path)
    data_handler.preprocess(base_path=framework_base_path,
                            min_seq_length=min_seq_length,
                            max_seq_length=max_seq_length)
    auto_eval_tasks = data_handler.get_tasks(base_path=framework_base_path,
                                             min_seq_length=min_seq_length,
                                             max_seq_length=max_seq_length,
                                             development_mode=development_mode)

    return auto_eval_tasks
