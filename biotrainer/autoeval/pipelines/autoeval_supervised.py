import h5py

from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from biotrainer_core.data_classes import Protocol, SequenceData
from biotrainer_core.functions.hashing import calculate_sequence_hash
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalProgress, SupervisedFrameworkReport

from typing import Optional, Callable, Dict, Tuple, List, Any, Generator

from ..core import AutoEvalFramework, AutoEvalConfigBank

from ...shared import get_device
from ...training.output_files import BiotrainerOutputObserver
from ...training.utilities.executer import parse_config_file_and_execute_run
from ...embedding import EmbeddingService, get_embedding_service, CustomEmbedder


class _PipelineEmbedder(ABC):

    @abstractmethod
    def per_residue_path(self, seqs: List[str]) -> Path:
        pass

    @abstractmethod
    def per_sequence_path(self, seqs: List[str]) -> Path:
        pass


class CustomEmbedderWrapper(_PipelineEmbedder):
    def __init__(self, custom_embedder: CustomEmbedder, output_path_per_res: Path, output_path_per_seq: Path):
        self.custom_embedder = custom_embedder
        self.output_path_per_res = output_path_per_res
        self.output_path_per_seq = output_path_per_seq

    @staticmethod
    def _wrap(embeddings_file_path: Path, sequences: List[str], custom_embedding_function: Callable):
        existing_hashes = set()
        if embeddings_file_path.is_file():
            with h5py.File(embeddings_file_path, "r") as embeddings_file:
                existing_hashes = set(embeddings_file.keys())

        # Filter out already existing sequences by hash
        seqs_to_compute = [seq for seq in sequences if calculate_sequence_hash(seq) not in existing_hashes]

        if len(seqs_to_compute) > 0:
            with h5py.File(embeddings_file_path, "a") as embeddings_file:
                idx = 0
                for sequence, embedding in custom_embedding_function(seqs_to_compute):
                    if len(embedding.shape) > 1 and embedding.shape[0] != len(sequence):
                        raise Exception(f"Per-residue embedding shape does not match sequence length - "
                                        f"Embedding Shape: {embedding.shape}, Sequence Length: {len(sequence)}!")
                    emb_record = SequenceData(seq_id=f"Seq{idx}", seq=sequence, embedding=embedding)
                    EmbeddingService.store_embedding(embeddings_file_handle=embeddings_file,
                                                     emb_record=emb_record,
                                                     store_by_hash=True)
                    idx += 1

        return embeddings_file_path

    def per_residue_path(self, seqs: List[str]) -> Path:
        return self._wrap(embeddings_file_path=self.output_path_per_res, sequences=seqs,
                          custom_embedding_function=self.custom_embedder.per_residue)

    def per_sequence_path(self, seqs: List[str]) -> Path:
        return self._wrap(embeddings_file_path=self.output_path_per_seq, sequences=seqs,
                          custom_embedding_function=self.custom_embedder.per_sequence)


class BuiltInEmbedder(_PipelineEmbedder):
    def __init__(self, embedder_name: str, output_dir: Path, use_half_precision: bool = False, device=None):
        self.embedder_name = embedder_name
        self.output_dir = output_dir
        self.use_half_precision = use_half_precision
        self.device = get_device(device)
        self.embedding_service: EmbeddingService = get_embedding_service(embedder_name=embedder_name,
                                                                         custom_tokenizer_config=None,
                                                                         use_half_precision=use_half_precision,
                                                                         device=get_device(device)
                                                                         )

    def per_residue_path(self, seqs: List[str]) -> Path:
        return Path(self.embedding_service.compute_embeddings(input_data=seqs,
                                                              output_dir=self.output_dir,
                                                              protocol=
                                                              Protocol.using_per_residue_embeddings()[
                                                                  0],
                                                              force_recomputing=False,
                                                              force_output_dir=True
                                                              ))

    def per_sequence_path(self, seqs: List[str]) -> Path:
        return Path(self.embedding_service.compute_embeddings(input_data=seqs,
                                                              output_dir=self.output_dir,
                                                              protocol=
                                                              Protocol.using_per_sequence_embeddings()[
                                                                  0],
                                                              force_recomputing=False,
                                                              force_output_dir=True
                                                              ))


class PreComputedEmbedder(_PipelineEmbedder):
    def __init__(self, per_residue_embeddings_path: Path, per_sequence_embeddings_path: Path):
        self.per_residue_embeddings_path = per_residue_embeddings_path
        self.per_sequence_embeddings_path = per_sequence_embeddings_path

    def per_residue_path(self, seqs: List[str]) -> Path:
        print(f"Using precomputed per-residue embeddings: {self.per_residue_embeddings_path}")
        return self.per_residue_embeddings_path

    def per_sequence_path(self, seqs: List[str]) -> Path:
        print(f"Using precomputed per-sequence embeddings: {self.per_sequence_embeddings_path}")
        return self.per_sequence_embeddings_path


def check_h5_file(name: str, h5_path: Optional[Path], expected_length: int) -> None:
    if h5_path is None:
        raise Exception(f"Did not find embeddings file for {name} after embedding calculation!")
    try:
        with h5py.File(h5_path, "r") as h5_file:
            actual_length = len([k for k in tqdm(h5_file.keys(), desc=f"Checking {name} embeddings..")])
            if actual_length < expected_length:
                raise ValueError(f"Expected {expected_length} entries in {name} h5 file but found {actual_length}!")
    except (OSError, IOError) as e:
        raise Exception(f"Could not read {name} h5 file: {str(e)}")


def setup_embedder(embedder_name,
                   output_dir,
                   use_half_precision: bool = False,
                   precomputed_per_residue_embeddings: Optional[Path] = None,
                   precomputed_per_sequence_embeddings: Optional[Path] = None,
                   custom_embedder: Optional[CustomEmbedder] = None,
                   device=None, ) -> _PipelineEmbedder:
    assert (precomputed_per_residue_embeddings is None) == (precomputed_per_sequence_embeddings is None)
    # Precomputed Embeddings -> Return paths
    if precomputed_per_residue_embeddings and precomputed_per_sequence_embeddings:
        return PreComputedEmbedder(per_residue_embeddings_path=precomputed_per_residue_embeddings,
                                   per_sequence_embeddings_path=precomputed_per_sequence_embeddings)

    # Custom embedder
    if custom_embedder:
        embeddings_file_path_per_residue = output_dir / EmbeddingService.get_embeddings_file_name(
            embedder_name=embedder_name,
            use_half_precision=use_half_precision,
            use_reduced_embeddings=False)
        embeddings_file_path_per_sequence = output_dir / EmbeddingService.get_embeddings_file_name(
            embedder_name=embedder_name,
            use_half_precision=use_half_precision,
            use_reduced_embeddings=True)
        return CustomEmbedderWrapper(custom_embedder=custom_embedder,
                                     output_path_per_res=embeddings_file_path_per_residue,
                                     output_path_per_seq=embeddings_file_path_per_sequence)

    # No custom embedding functions -> Biotrainer Embedding Service
    return BuiltInEmbedder(embedder_name=embedder_name,
                           output_dir=output_dir,
                           use_half_precision=use_half_precision,
                           device=get_device(device))


def autoeval_supervised_pipeline(embedder_name: str,
                                 framework: AutoEvalFramework,
                                 embeddings_file_per_sequence: Optional[Path],
                                 embeddings_file_per_residue: Optional[Path],
                                 task_config_tuples: List[Tuple[AutoEvalTask, Dict[str, Any]]],
                                 output_dir: Path,
                                 framework_report: SupervisedFrameworkReport,
                                 custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None,
                                 device=None,
                                 ) -> Generator[AutoEvalProgress, None, None]:
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
        maybe_result = framework_report.maybe_load_existing_result(embedder_name=embedder_name,
                                                                   task_output_dir=task_output_dir)
        if maybe_result:
            print(f"Loaded existing result for task {current_task_name}, skipping execution..")
            framework_report.update_result(combined_task_name=current_task_name, result=maybe_result)
            continue

        # No result exists yet -> execute biotrainer
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
                                                   custom_output_observers=custom_output_observers)

        framework_report.update_result(combined_task_name=current_task_name, result=result)

        completed_tasks += 1
        print(f"Finished task execution for {current_task_name}!")

    print(f"Autoeval supervised pipeline on framework {framework.get_name()} "
          f"for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=framework_report)
