import os
import h5py
import torch

from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, List, Any, Union, Iterable, Generator

from .autoeval_setup import setup_pipeline
from .autoeval_validate import validate_input
from .autoeval_progress import AutoEvalProgress
from ..autoeval_frameworks import AvailableFramework
from .autoeval_report import AutoEvalReport, SupervisedFrameworkReport
from ..core import AutoEvalFramework, AutoEvalConfigBank, AutoEvalTask

from ...trainers import Pipeline
from ...protocols import Protocol
from ...utilities import get_device
from ...output_files import BiotrainerOutputObserver
from ...input_files import read_FASTA, BiotrainerSequenceRecord
from ...embedders import EmbeddingService, get_embedding_service
from ...utilities.executer import parse_config_file_and_execute_run


def get_unique_framework_sequences(framework: Union[str, AvailableFramework, AutoEvalFramework],
                                   min_seq_length: int,
                                   max_seq_length: int,
                                   custom_storage_path: Optional[str] = None,
                                   force_download: Optional[bool] = False
                                   ) -> Tuple[
    List[Tuple[AutoEvalTask, Dict[str, Any]]], Dict[str, BiotrainerSequenceRecord],
    Dict[str, BiotrainerSequenceRecord]]:
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
                                     force_download=force_download)
    task_config_tuples = []
    for task in auto_eval_tasks:
        config = config_bank.get_task_config(task=task)
        task_config_tuples.append((task, config))
        input_files = task.input_files
        if len(input_files) != 1:
            # Only one input file per supervised task
            raise AssertionError(f"Expected exactly one input file for task "
                                 f"{task.combined_name()} but found {len(input_files)}!")

    unique_per_residue, unique_per_sequence = _get_unique_sequences_for_all_tasks(
        {str(t.input_files[0]): Protocol.from_string(c["protocol"]) for t, c in task_config_tuples}
    )
    return task_config_tuples, unique_per_residue, unique_per_sequence


def _get_unique_sequences_for_all_tasks(tasks: Dict[str, Protocol]) -> Tuple[Dict[str,
BiotrainerSequenceRecord], Dict[str, BiotrainerSequenceRecord]]:
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


def _check_h5_file(name: str, h5_path: Optional[Path], expected_length: int) -> None:
    if h5_path is None:
        raise Exception(f"Did not find embeddings file for {name} after embedding calculation!")
    try:
        with h5py.File(h5_path, "r") as h5_file:
            actual_length = len(h5_file.keys())
            if actual_length != expected_length:
                raise ValueError(f"Expected {expected_length} entries in {name} h5 file but found {actual_length}!")
    except (OSError, IOError) as e:
        raise Exception(f"Could not read {name} h5 file: {str(e)}")


def _run_pipeline(embedder_name: str,
                  framework: AutoEvalFramework,
                  autoeval_report: AutoEvalReport,
                  embedding_function_per_residue: Optional[Callable[[Iterable[str]], Path]],
                  embedding_function_per_sequence: Optional[Callable[[Iterable[str]], Path]],
                  output_dir: Path,
                  min_seq_length: int,
                  max_seq_length: int,
                  custom_pipeline: Optional[Pipeline] = None,
                  custom_storage_path: Optional[str] = None,
                  custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None,
                  force_download: Optional[bool] = False,
                  device = None,
                  ) -> Generator[AutoEvalProgress, None, None]:
    task_config_tuples, unique_per_residue, unique_per_sequence = get_unique_framework_sequences(framework=framework,
                                                                                                 min_seq_length=min_seq_length,
                                                                                                 max_seq_length=max_seq_length,
                                                                                                 custom_storage_path=custom_storage_path,
                                                                                                 force_download=force_download)
    # Embed if no custom pipeline provided - that must handle the embedding step independently
    embeddings_file_per_residue = None
    embeddings_file_per_sequence = None
    if not custom_pipeline:
        print(f"Embedding {len(unique_per_residue)} sequences per_residue")
        embeddings_file_per_residue = embedding_function_per_residue(
            [seq_record.seq for _, seq_record in unique_per_residue.items()]
        )

        print(f"Embedding {len(unique_per_sequence)} sequences per_sequence")
        embeddings_file_per_sequence = embedding_function_per_sequence(
            [seq_record.seq for _, seq_record in unique_per_sequence.items()]
        )

    _check_h5_file(name="per-residue", h5_path=embeddings_file_per_residue, expected_length=len(unique_per_residue))
    _check_h5_file(name="per-sequence", h5_path=embeddings_file_per_sequence, expected_length=len(unique_per_sequence))

    print("Calculated embeddings successfully!")

    # Framework results do not exist yet -> execute biotrainer
    supervised_framework_report = SupervisedFrameworkReport.empty(min_seq_len=min_seq_length,
                                                                  max_seq_len=max_seq_length)
    task_names = [task.combined_name() for task, _ in task_config_tuples]
    print(f"The following tasks will be executed in order: {task_names} (total {len(task_names)})")
    completed_tasks = 0
    total_tasks = len(task_config_tuples)
    current_task_name = ""
    for task, config in task_config_tuples:
        current_task_name = task.combined_name()
        print(f"Running task {current_task_name}...")
        yield AutoEvalProgress(completed_tasks=completed_tasks, total_tasks=total_tasks,
                               current_task_name=current_task_name,
                               current_framework_name=framework.get_name())

        task_output_dir = output_dir / current_task_name
        # Check if result already exists -> skip (Framework run was interrupted)
        maybe_result = supervised_framework_report.maybe_load_existing_result(embedder_name=embedder_name,
                                                                              task_output_dir=task_output_dir)
        if maybe_result:
            print(f"Loaded existing result for task {current_task_name}, skipping execution..")
            supervised_framework_report.update_result(combined_task_name=current_task_name, result=maybe_result)
            continue

        # No result exists yet -> execute biotrainer
        if custom_pipeline:
            task_embeddings_file = None
        else:
            assert embeddings_file_per_sequence is not None and embeddings_file_per_residue is not None
            task_embeddings_file = embeddings_file_per_sequence if (Protocol.from_string(config["protocol"]) in
                                                                    Protocol.using_per_sequence_embeddings()) \
                else embeddings_file_per_residue
        config = AutoEvalConfigBank.add_custom_values_to_config(config=config,
                                                                embedder_name=embedder_name,
                                                                embeddings_file=task_embeddings_file,
                                                                input_file=task.input_files[0],
                                                                output_dir=task_output_dir,
                                                                device=device,
                                                                )

        result = parse_config_file_and_execute_run(config=config,
                                                   custom_pipeline=custom_pipeline,
                                                   custom_output_observers=custom_output_observers)

        supervised_framework_report.update_result(combined_task_name=current_task_name, result=result)

        completed_tasks += 1
        print(f"Finished task execution for {current_task_name}!")

    autoeval_report.add_supervised_result(framework_name=framework.get_name(), report=supervised_framework_report)
    autoeval_report.write(output_dir=output_dir.parent)

    print(f"Autoeval pipeline on framework {framework.get_name()} for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=autoeval_report)


def _setup_embedding_functions(embedder_name,
                               output_dir,
                               use_half_precision: Optional[bool] = False,
                               custom_pipeline: Optional[Pipeline] = None,
                               precomputed_per_residue_embeddings: Optional[Path] = None,
                               precomputed_per_sequence_embeddings: Optional[Path] = None,
                               custom_tokenizer_config: Optional[dict] = None,
                               custom_embedding_function_per_residue: Optional[
                                   Callable[[Iterable[str]], Generator[Tuple[str, torch.tensor], None, None]]] = None,
                               custom_embedding_function_per_sequence: Optional[
                                   Callable[
                                       [Iterable[str]], Generator[Tuple[str, torch.tensor], None, None]]] = None,
                               device=None, ):
    # Custom Pipeline -> Embedding calculation handled inside of pipeline
    if custom_pipeline:
        return None, None

    # Precomputed Embeddings -> Return paths
    assert (precomputed_per_residue_embeddings is None) == (precomputed_per_sequence_embeddings is None)
    if precomputed_per_residue_embeddings and precomputed_per_sequence_embeddings:
        def precomputed_per_res(seqs):
            print(f"Using precomputed per-residue embeddings: {precomputed_per_residue_embeddings}")
            return precomputed_per_residue_embeddings

        def precomputed_per_seq(seqs):
            print(f"Using precomputed per-sequence embeddings: {precomputed_per_sequence_embeddings}")
            return precomputed_per_sequence_embeddings

        return precomputed_per_res, precomputed_per_seq

    # No custom embedding functions -> Biotrainer Embedding Service
    assert (custom_embedding_function_per_residue is None) == (custom_embedding_function_per_sequence is None)
    if not custom_embedding_function_per_residue and not custom_embedding_function_per_sequence:
        embedding_service: EmbeddingService = get_embedding_service(embedder_name=embedder_name,
                                                                    custom_tokenizer_config=custom_tokenizer_config,
                                                                    use_half_precision=use_half_precision,
                                                                    device=get_device(device)
                                                                    )
        embedding_function_per_residue = lambda seqs: embedding_service.compute_embeddings(input_data=seqs,
                                                                                           output_dir=output_dir,
                                                                                           protocol=
                                                                                           Protocol.using_per_residue_embeddings()[
                                                                                               0],
                                                                                           force_recomputing=False,
                                                                                           force_output_dir=True
                                                                                           )
        embedding_function_per_sequence = lambda seqs: embedding_service.compute_embeddings(input_data=seqs,
                                                                                            output_dir=output_dir,
                                                                                            protocol=
                                                                                            Protocol.using_per_sequence_embeddings()[
                                                                                                0],
                                                                                            force_recomputing=False,
                                                                                            force_output_dir=True
                                                                                            )
        return embedding_function_per_residue, embedding_function_per_sequence

    # Custom embedding functions -> Use wrapper
    embeddings_file_path_per_residue = output_dir / EmbeddingService.get_embeddings_file_name(
        embedder_name=embedder_name,
        use_half_precision=use_half_precision,
        use_reduced_embeddings=False)
    if os.path.exists(embeddings_file_path_per_residue):
        print("Using existing embeddings file per-residue!")
        embedding_function_per_residue = lambda seqs: embeddings_file_path_per_residue
    else:
        embedding_function_per_residue = lambda seqs: _wrap_custom_embedding_function(
            custom_embedding_function_per_residue,
            embeddings_file_path_per_residue,
            seqs)
    embeddings_file_path_per_sequence = output_dir / EmbeddingService.get_embeddings_file_name(
        embedder_name=embedder_name,
        use_half_precision=use_half_precision,
        use_reduced_embeddings=True)
    if os.path.exists(embeddings_file_path_per_sequence):
        print("Using existing embeddings file per-sequence!")
        embedding_function_per_sequence = lambda seqs: embeddings_file_path_per_sequence
    else:
        embedding_function_per_sequence = lambda seqs: _wrap_custom_embedding_function(
            custom_embedding_function_per_sequence,
            embeddings_file_path_per_sequence,
            seqs)

    return embedding_function_per_residue, embedding_function_per_sequence


def _wrap_custom_embedding_function(
        custom_embedding_function: Callable[[Iterable[str]], Generator[Tuple[str, torch.tensor], None, None]],
        embeddings_file_path: Path,
        sequences: Iterable[str],
):
    with h5py.File(embeddings_file_path, "a") as embeddings_file:
        idx = 0
        for sequence, embedding in custom_embedding_function(sequences):
            if len(embedding.shape) > 1 and embedding.shape[0] != len(sequence):
                raise Exception(f"Per-residue embedding shape does not match sequence length - "
                                f"Embedding Shape: {embedding.shape}, Sequence Length: {len(sequence)}!")
            seq_record = BiotrainerSequenceRecord(seq_id=f"Seq{idx}", seq=sequence)
            EmbeddingService.store_embedding(embeddings_file_handle=embeddings_file,
                                             seq_record=seq_record,
                                             embedding=embedding,
                                             store_by_hash=True)
            idx += 1

    return embeddings_file_path


def autoeval_supervised_pipeline(embedder_name: str,
                                 framework: AutoEvalFramework,
                                 autoeval_report: AutoEvalReport,
                                 output_dir: Optional[Union[Path, str]] = "autoeval_output",
                                 force_download: Optional[bool] = False,
                                 use_half_precision: Optional[bool] = False,
                                 min_seq_length: Optional[int] = 0,
                                 max_seq_length: Optional[int] = 2000,
                                 custom_pipeline: Optional[Pipeline] = None,
                                 custom_tokenizer_config: Optional[dict] = None,
                                 precomputed_per_residue_embeddings: Optional[Path] = None,
                                 precomputed_per_sequence_embeddings: Optional[Path] = None,
                                 custom_embedding_function_per_residue: Optional[
                                     Callable[[Iterable[str]], Generator[Tuple[str, torch.tensor], None, None]]] = None,
                                 custom_embedding_function_per_sequence: Optional[
                                     Callable[[Iterable[str]], Generator[Tuple[str, torch.tensor], None, None]]] = None,
                                 custom_storage_path: Optional[Union[Path, str]] = None,
                                 custom_output_observers: List[BiotrainerOutputObserver] = None,
                                 device=None,
                                 ) -> Generator[AutoEvalProgress, None, None]:
    if custom_pipeline is not None and any([v is not None for v in [custom_tokenizer_config,
                                                                    precomputed_per_residue_embeddings,
                                                                    precomputed_per_sequence_embeddings,
                                                                    custom_embedding_function_per_residue,
                                                                    custom_embedding_function_per_sequence]]):
        raise ValueError(f"You must either provide a custom_pipeline or custom embedding functions and configurations!")

    if (precomputed_per_residue_embeddings is None) ^ (precomputed_per_sequence_embeddings is None):
        raise ValueError(f"You must provide either paths to both precomputed per-sequence and per-residue embeddings "
                         f"or no precomputed path at all!")
    using_precomputed_embeddings = precomputed_per_residue_embeddings is not None and precomputed_per_sequence_embeddings is not None

    if (custom_embedding_function_per_residue is None) ^ (custom_embedding_function_per_sequence is None):
        raise ValueError(f"You must provide either a custom embedding function for per-sequence and per-residue or no "
                         f"custom function at all!")

    using_custom_embedding_functions = custom_embedding_function_per_residue is not None and custom_embedding_function_per_sequence is not None

    if using_precomputed_embeddings and using_custom_embedding_functions:
        raise ValueError(f"You must either provide precomputed embeddings or custom embedding functions, not both!")

    # Setup
    embedding_function_per_residue, embedding_function_per_sequence = _setup_embedding_functions(
        embedder_name=embedder_name,
        output_dir=output_dir,
        use_half_precision=use_half_precision,
        custom_pipeline=custom_pipeline,
        custom_tokenizer_config=custom_tokenizer_config,
        precomputed_per_residue_embeddings=precomputed_per_residue_embeddings,
        precomputed_per_sequence_embeddings=precomputed_per_sequence_embeddings,
        custom_embedding_function_per_residue=custom_embedding_function_per_residue,
        custom_embedding_function_per_sequence=custom_embedding_function_per_sequence,
        device=device)

    # Pipeline
    yield from _run_pipeline(embedder_name=embedder_name,
                             framework=framework,
                             autoeval_report=autoeval_report,
                             embedding_function_per_residue=embedding_function_per_residue,
                             embedding_function_per_sequence=embedding_function_per_sequence,
                             output_dir=output_dir,
                             min_seq_length=min_seq_length,
                             max_seq_length=max_seq_length,
                             custom_pipeline=custom_pipeline,
                             custom_storage_path=custom_storage_path,
                             custom_output_observers=custom_output_observers,
                             force_download=force_download,
                             device=device,
                             )
