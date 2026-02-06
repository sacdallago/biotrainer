import torch

from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Tuple, List, Union, Iterable, Generator

from .core import AutoEvalFramework, AutoEvalMode
from .pipelines import (AutoEvalReport, setup_output_dir, validate_input, autoeval_supervised_pipeline,
                        autoeval_zeroshot_pipeline, AutoEvalProgress)
from .autoeval_frameworks import AvailableFramework

from ..trainers import Pipeline
from ..bioengineer import ZeroShotMethod
from ..output_files import BiotrainerOutputObserver


def autoeval_pipeline(embedder_name: str,
                      framework: Union[str, AvailableFramework],
                      zero_shot_method: Optional[ZeroShotMethod] = None,
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
                      ) -> Generator[AutoEvalProgress, None, None]:
    """
    Run the autoeval pipeline for a given embedder_name and framework.

    :param embedder_name: The name of the embedder. Usually a huggingface pretrained embedder in format org/embed_name.
    :param framework: The framework to be evaluated. Currently, only FLIP is available.
    :param zero_shot_method: The zero-shot method to use. Only for zero-shot framework evaluation.
    :param output_dir: The directory to save the output to, defaults to "autoeval_output".
    :param force_download: Flag to determine whether to force re-downloading the framework datasets, defaults to False.
    :param use_half_precision: Flag to determine whether to use a half-precision floating point for the embedder or not.
    :param min_seq_length: The minimum sequence length to pre-filter the framework datasets, defaults to 0.
    :param max_seq_length: The maximum sequence length to pre-filter the framework datasets, defaults to 2000.
    :param custom_pipeline: Custom pipeline to be executed, defaults to None (= default biotrainer pipeline).
        If a custom pipeline is specified, no other custom parameters for embedding must be provided. The pipeline
        must handle embeddings on its own.
    :param custom_tokenizer_config: Custom tokenizer configuration dictionary for onnx models.
    :param precomputed_per_residue_embeddings:
        Optional path to precomputed per-residue embeddings.
        Must be provided together with per-sequence embeddings path.
        The embeddings must be stored by sequence hash in a .h5 file. Lear more here: docs/h5_file_standardization.md
    :param precomputed_per_sequence_embeddings:
        Optional path to precomputed per-sequence embeddings.
        Must be provided together with per-residue embeddings path.
        The embeddings must be stored by sequence hash in a .h5 file. Lear more here: docs/h5_file_standardization.md
    :param custom_embedding_function_per_residue:
        Custom per-residue embedding function that is used instead
        of the biotrainer embedding service if provided.
        Takes an iterable of sequence strings as input and must provide the per-residue embeddings as a generator.
    :param custom_embedding_function_per_sequence:
        Custom per-sequence embedding function that is used instead
        of the biotrainer embedding service if provided.
        Takes an iterable of sequence strings as input and must provide the per-sequence embeddings as a generator.
    :param custom_storage_path: Optional path where to store the framework datasets if not downloaded yet.
    :param custom_output_observers: Optional list of custom training output observers.
    :return: A dictionary containing the autoeval pipeline results. Each task result is a biotrainer model output dict.
    """
    framework_obj: AutoEvalFramework = validate_input(framework,
                                                      zero_shot_method=zero_shot_method,
                                                      min_seq_length=min_seq_length,
                                                      max_seq_length=max_seq_length)

    if force_download and custom_storage_path:
        raise ValueError(f"Cannot force download and use custom storage path at the same time!"
                         f"force_download only clears the cache directory, "
                         f"so it is not necessary when using custom_storage_path, "
                         f"just make sure that that is up-to-date.")

    # Setup
    output_dir = setup_output_dir(base_dir=output_dir,
                                  embedder_name=embedder_name,
                                  framework_name=framework_obj.get_name())
    # Check if results already exist
    autoeval_report = AutoEvalReport.loaded_or_empty(embedder_name=embedder_name,
                                                     training_date=str(datetime.now().date().isoformat()),
                                                     output_dir=output_dir.parent)
    # Framework results already exist -> skip execution
    maybe_framework_result = autoeval_report.maybe_framework_result(framework_name=framework_obj.get_name())
    if maybe_framework_result:
        print(f"Autoeval report for framework {framework_obj.get_name()} already exists, skipping execution!")
        yield AutoEvalProgress(completed_tasks=maybe_framework_result.number_tasks(),
                               total_tasks=maybe_framework_result.number_tasks(),
                               current_task_name="",
                               current_framework_name=framework_obj.get_name(),
                               final_report=autoeval_report)
        return

    # Framework results do not exist yet -> execute autoeval pipeline
    match framework_obj.get_mode():
        case AutoEvalMode.SUPERVISED:
            yield from autoeval_supervised_pipeline(embedder_name=embedder_name,
                                                    framework=framework_obj,
                                                    autoeval_report=autoeval_report,
                                                    output_dir=output_dir,
                                                    force_download=force_download,
                                                    use_half_precision=use_half_precision,
                                                    min_seq_length=min_seq_length,
                                                    max_seq_length=max_seq_length,
                                                    custom_pipeline=custom_pipeline,
                                                    custom_tokenizer_config=custom_tokenizer_config,
                                                    precomputed_per_residue_embeddings=precomputed_per_residue_embeddings,
                                                    precomputed_per_sequence_embeddings=precomputed_per_sequence_embeddings,
                                                    custom_embedding_function_per_residue=custom_embedding_function_per_residue,
                                                    custom_embedding_function_per_sequence=custom_embedding_function_per_sequence,
                                                    custom_storage_path=custom_storage_path,
                                                    custom_output_observers=custom_output_observers)
        case AutoEvalMode.ZERO_SHOT:
            yield from autoeval_zeroshot_pipeline(embedder_name=embedder_name,
                                                  framework=framework_obj,
                                                  method=zero_shot_method,
                                                  autoeval_report=autoeval_report,
                                                  output_dir=output_dir,
                                                  force_download=force_download,
                                                  custom_storage_path=custom_storage_path)
