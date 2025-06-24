import os

from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, Tuple, List, Any, Union, Iterable

from .report_manager import ReportManager
from .config_bank import AutoEvalConfigBank
from .flip import FLIPDataHandler, FLIPConfigBank
from .data_handler import AutoEvalDataHandler, AutoEvalTask

from ..trainers import Pipeline
from ..protocols import Protocol
from ..utilities import get_device
from ..output_files import BiotrainerOutputObserver
from ..input_files import read_FASTA, BiotrainerSequenceRecord
from ..embedders import EmbeddingService, get_embedding_service
from ..utilities.executer import parse_config_file_and_execute_run


def available_frameworks() -> Dict[str, Tuple[AutoEvalDataHandler, AutoEvalConfigBank]]:
    return {"flip": (FLIPDataHandler(), FLIPConfigBank())}


def _framework_factory(framework: str):
    return available_frameworks()[framework]


def _validate_input(framework: str, min_seq_length: int, max_seq_length: int) -> None:
    if framework not in available_frameworks():
        raise ValueError(f"Unsupported framework: {framework}")

    if min_seq_length >= max_seq_length:
        raise ValueError("min_seq_length must be less than max_seq_length")

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be greater than 0")


def get_unique_framework_sequences(framework: str,
                                   min_seq_length: int,
                                   max_seq_length: int,
                                   custom_framework_storage_path: Optional[str] = None,
                                   ) -> (List[Tuple[AutoEvalTask, Dict[str, Any]]],
                                         Dict[str, BiotrainerSequenceRecord],
                                         Dict[str, BiotrainerSequenceRecord]
                                         ):
    _validate_input(framework, min_seq_length, max_seq_length)

    data_handler, config_bank = _framework_factory(framework)
    auto_eval_tasks = _setup_pipeline(data_handler=data_handler,
                                      min_seq_length=min_seq_length,
                                      max_seq_length=max_seq_length,
                                      custom_framework_storage_path=custom_framework_storage_path)
    task_config_tuples = []
    for task in auto_eval_tasks:
        config = config_bank.get_task_config(task=task)
        task_config_tuples.append((task, config))

    unique_per_residue, unique_per_sequence = _get_unique_sequences_for_all_tasks(
        {str(t.input_file): Protocol.from_string(c["protocol"]) for t, c in task_config_tuples}
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


def _setup_pipeline(data_handler: AutoEvalDataHandler,
                    min_seq_length: int,
                    max_seq_length: int,
                    custom_framework_storage_path: Optional[str] = None,
                    ) -> List[AutoEvalTask]:
    framework_base_path = data_handler.get_framework_base_path(
        custom_framework_storage_path=custom_framework_storage_path)

    if not os.path.exists(framework_base_path):
        os.makedirs(framework_base_path, exist_ok=True)

    if data_handler.is_download_necessary(framework_base_path):
        data_handler.download_data(data_dir=framework_base_path)
    data_handler.preprocess(base_path=framework_base_path,
                            min_seq_length=min_seq_length,
                            max_seq_length=max_seq_length)
    auto_eval_tasks = data_handler.get_tasks(base_path=framework_base_path, min_seq_length=min_seq_length,
                                             max_seq_length=max_seq_length)

    return auto_eval_tasks


def _run_pipeline(embedder_name: str,
                  framework: str,
                  embedding_function_per_residue: Callable[[Iterable[str]], Path],
                  embedding_function_per_sequence: Callable[[Iterable[str]], Path],
                  output_dir: Path,
                  min_seq_length: int,
                  max_seq_length: int,
                  custom_pipeline: Optional[Pipeline] = None,
                  custom_framework_storage_path: Optional[str] = None,
                  custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None,
                  ) -> Dict[str, Any]:
    task_config_tuples, unique_per_residue, unique_per_sequence = get_unique_framework_sequences(framework=framework,
                                                                                                 min_seq_length=min_seq_length,
                                                                                                 max_seq_length=max_seq_length,
                                                                                                 custom_framework_storage_path=custom_framework_storage_path)
    # A custom pipeline must handle the embedding step independently
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

        print("Calculated embeddings successfully!")

    report_manager = ReportManager(embedder_name=embedder_name,
                                   training_date=str(datetime.now().date().isoformat())
                                   )
    # Execute biotrainer
    for task, config in task_config_tuples:
        print(f"Running task {task.name}...")
        task_output_dir = output_dir / task.name
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
                                                                input_file=task.input_file,
                                                                output_dir=task_output_dir,
                                                                )
        result = parse_config_file_and_execute_run(config=config,
                                                   custom_pipeline=custom_pipeline,
                                                   custom_output_observers=custom_output_observers)

        report_manager.add_result(task=task, result_dict=result)
        print(f"Finished task {task.name}!")

    report = report_manager.write(output_dir=output_dir)

    print(f"Autoeval pipeline for {embedder_name} finished successfully!")
    return report


def autoeval_pipeline(embedder_name: str,
                      framework: str,
                      output_dir: Optional[Union[Path, str]] = "autoeval_output",
                      use_half_precision: Optional[bool] = False,
                      min_seq_length: Optional[int] = 0,
                      max_seq_length: Optional[int] = 2000,
                      custom_pipeline: Optional[Pipeline] = None,
                      custom_tokenizer_config: Optional[dict] = None,
                      custom_embedding_function_per_residue: Optional[Callable[[Iterable[str]], Path]] = None,
                      custom_embedding_function_per_sequence: Optional[Callable[[Iterable[str]], Path]] = None,
                      custom_framework_storage_path: Optional[Union[Path, str]] = None,
                      custom_output_observers: List[BiotrainerOutputObserver] = None,
                      ) -> Dict[str, Any]:
    """
    Run the autoeval pipeline for given embedder_name and framework.
    
    :param embedder_name: The name of the embedder. Usually a huggingface pretrained embedder in format org/embed_name.
    :param framework: The framework to be evaluated. Currently, only FLIP is available.
    :param output_dir: The directory to save the output to, defaults to "autoeval_output".
    :param use_half_precision: Flag to determine whether to use a half-precision floating point for the embedder or not.
    :param min_seq_length: The minimum sequence length to pre-filter the framework datasets, defaults to 0.
    :param max_seq_length: The maximum sequence length to pre-filter the framework datasets, defaults to 2000.
    :param custom_pipeline: Custom pipeline to be executed, defaults to None (= default biotrainer pipeline).
        If a custom pipeline is specified, no other custom parameters for embedding must be provided. The pipeline
        must handle embeddings on its own.
    :param custom_tokenizer_config: Custom tokenizer configuration dictionary for onnx models.
    :param custom_embedding_function_per_residue: 
        Custom per-residue embedding function that is used instead 
        of the biotrainer embedding service if provided.
        Takes an iterable of sequence strings as input and must provide the 
        full path to the saved per-residue embeddings as an output.
    :param custom_embedding_function_per_sequence: 
        Custom per-sequence embedding function that is used instead 
        of the biotrainer embedding service if provided.
        Takes an iterable of sequence strings as input and must provide the 
        full path to the saved per-sequence embeddings as an output.
    :param custom_framework_storage_path: Optional path where to store the framework datasets if not downloaded yet.
    :param custom_output_observers: Optional list of custom training output observers.
    :return: A dictionary containing the autoeval pipeline results. Each task result is a biotrainer model output dict.
    """
    _validate_input(framework, min_seq_length, max_seq_length)

    if custom_pipeline is not None and any([v is not None for v in [custom_tokenizer_config,
                                                                    custom_embedding_function_per_residue,
                                                                    custom_framework_storage_path]]):
        raise ValueError(f"You must either provide a custom_pipeline or custom embedding functions and configurations!")

    # Setup 
    embedder_dir_name = embedder_name
    if "/" in embedder_dir_name:  # Huggingface
        embedder_dir_name = embedder_dir_name.replace("/", "-")

    output_dir = Path(output_dir) / embedder_dir_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup embedding functions
    if not custom_pipeline:
        embedding_service: Optional[EmbeddingService] = None
        if not custom_embedding_function_per_residue or not custom_embedding_function_per_sequence:
            embedding_service: EmbeddingService = get_embedding_service(embedder_name=embedder_name,
                                                                        custom_tokenizer_config=custom_tokenizer_config,
                                                                        use_half_precision=use_half_precision,
                                                                        device=get_device()
                                                                        )

        if not custom_embedding_function_per_residue:
            assert embedding_service is not None
            custom_embedding_function_per_residue = lambda seqs: embedding_service.compute_embeddings(input_data=seqs,
                                                                                                      output_dir=output_dir,
                                                                                                      protocol=
                                                                                                      Protocol.using_per_residue_embeddings()[
                                                                                                          0],
                                                                                                      force_recomputing=False,
                                                                                                      force_output_dir=True
                                                                                                      )

        if not custom_embedding_function_per_sequence:
            assert embedding_service is not None
            custom_embedding_function_per_sequence = lambda seqs: embedding_service.compute_embeddings(input_data=seqs,
                                                                                                       output_dir=output_dir,
                                                                                                       protocol=
                                                                                                       Protocol.using_per_sequence_embeddings()[
                                                                                                           0],
                                                                                                       force_recomputing=False,
                                                                                                       force_output_dir=True
                                                                                                       )

    return _run_pipeline(embedder_name=embedder_name,
                         framework=framework,
                         embedding_function_per_residue=custom_embedding_function_per_residue,
                         embedding_function_per_sequence=custom_embedding_function_per_sequence,
                         output_dir=output_dir,
                         min_seq_length=min_seq_length,
                         max_seq_length=max_seq_length,
                         custom_pipeline=custom_pipeline,
                         custom_framework_storage_path=custom_framework_storage_path
                         )
