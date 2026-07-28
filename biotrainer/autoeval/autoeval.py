from __future__ import annotations

import torch

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from biotrainer_core.data_classes import ZeroShotMethod
from typing import Optional, Callable, Tuple, List, Union, Generator, Dict, Any
from biotrainer_core.data_classes.autoeval import AutoEvalProgress, AutoEvalReport, FrameworkReport, AutoEvalTask

from .core import AutoEvalFramework
from .pipelines import (setup_output_dir, validate_input, autoeval_supervised_pipeline,
                        autoeval_zeroshot_pipeline, autoeval_zeroshot_contact_pipeline,
                        autoeval_supervised_contact_pipeline, get_unique_framework_sequences,
                        check_h5_file, setup_embedder)
from .autoeval_frameworks import AvailableFramework

from ..shared import get_device
from ..bioengineer import BioEngineer
from ..embedding import CustomEmbedder, EmbeddingService
from ..training.output_files import BiotrainerOutputObserver


@dataclass
class _AutoEvalTaskRunnerParams:
    task_config_tuples: List[Tuple[AutoEvalTask, Dict[str, Any]]]
    embeddings_file_per_residue: Optional[Path]
    embeddings_file_per_sequence: Optional[Path]
    device: Optional[Any]


@dataclass
class _AutoEvalTaskRunner:
    framework: AutoEvalFramework
    runner: Callable[[_AutoEvalTaskRunnerParams], Generator[AutoEvalProgress, None, None]]


class AutoEval:
    def __init__(self,
                 embedder_name: str,
                 output_dir: Union[Path, str] = "autoeval_output",
                 force_download: bool = False,
                 use_half_precision: bool = False,
                 min_seq_length: int = 0,
                 max_seq_length: int = 2000,
                 custom_storage_path: Optional[Union[Path, str]] = None,
                 precomputed_per_residue_embeddings: Optional[Path] = None,
                 precomputed_per_sequence_embeddings: Optional[Path] = None,
                 custom_embedder: Optional[CustomEmbedder] = None,
                 custom_bioengineer: Optional[BioEngineer] = None,
                 development_mode: bool = True,
                 ):
        """
        Initialize the AutoEval class for evaluating embedders across multiple frameworks.

        :param embedder_name: The name of the embedder. Usually a HuggingFace pretrained embedder 
            in format 'org/embed_name'.
        :param output_dir: The directory to save the output to. Defaults to "autoeval_output".
        :param force_download: Flag to determine whether to force re-downloading the framework datasets. 
            Defaults to False. Cannot be used together with custom_storage_path.
        :param use_half_precision: Flag to determine whether to use half-precision floating point 
            (float16) for the embedder. Defaults to False.
        :param min_seq_length: The minimum sequence length to pre-filter the framework datasets. 
            Defaults to 0.
        :param max_seq_length: The maximum sequence length to pre-filter the framework datasets. 
            Defaults to 2000.
        :param custom_storage_path: Optional path where to store the framework datasets if not 
            downloaded yet. Cannot be used together with force_download.
        :param precomputed_per_residue_embeddings: Optional path to precomputed per-residue embeddings.
            Must be provided together with precomputed_per_sequence_embeddings.
            The embeddings must be stored by sequence hash in a .h5 file. 
            Learn more here: docs/h5_file_standardization.md
        :param precomputed_per_sequence_embeddings: Optional path to precomputed per-sequence embeddings.
            Must be provided together with precomputed_per_residue_embeddings.
            The embeddings must be stored by sequence hash in a .h5 file. 
            Learn more here: docs/h5_file_standardization.md
        :param custom_embedder: Optional custom embedder instance that provides custom embedding functions.
            If provided, it is used instead of the biotrainer embedding service.
            Cannot be used together with precomputed embeddings.
        :param development_mode: If True, only subsets of the data are used for evaluation. Useful for fast development,
            and to have a separate evaluation run on the full dataset.
        :param custom_bioengineer: Optional custom bioengineer instance to use for zero-shot evaluation.
        :raises ValueError: If force_download and custom_storage_path are both provided.
        :raises ValueError: If only one of the precomputed embedding paths is provided.
        :raises ValueError: If both precomputed embeddings and custom_embedder are provided.
        """
        if force_download and custom_storage_path:
            raise ValueError(f"Cannot force download and use custom storage path at the same time!"
                             f"force_download only clears the cache directory, "
                             f"so it is not necessary when using custom_storage_path, "
                             f"just make sure that that is up-to-date.")
        if (precomputed_per_residue_embeddings is None) ^ (precomputed_per_sequence_embeddings is None):
            raise ValueError(
                f"You must provide either paths to both precomputed per-sequence and per-residue embeddings "
                f"or no precomputed path at all!")
        using_precomputed_embeddings = precomputed_per_residue_embeddings is not None and precomputed_per_sequence_embeddings is not None

        using_custom_embedding_functions = custom_embedder is not None

        if using_precomputed_embeddings and using_custom_embedding_functions:
            raise ValueError(f"You must either provide precomputed embeddings or custom embedding functions, not both!")

        self.output_dir = Path(output_dir)
        self.embedder_name = embedder_name
        self.force_download = force_download
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.use_half_precision = use_half_precision
        self.custom_storage_path = custom_storage_path
        self.development_mode = development_mode

        # Embeddings
        self.precomputed_per_residue_embeddings = precomputed_per_residue_embeddings
        self.precomputed_per_sequence_embeddings = precomputed_per_sequence_embeddings
        self.custom_embedder = custom_embedder

        # Bioengineer
        self.bioengineer = custom_bioengineer

        # Results
        self.autoeval_report: Optional[AutoEvalReport] = None
        self._frameworks_to_runners: Dict[AutoEvalFramework, Optional[_AutoEvalTaskRunner]] = {}
        self._results: Dict[AutoEvalFramework, FrameworkReport] = {}

    def _setup_runner_params(self, devices: Optional[List[Union[str, torch.device]]] = None) -> Dict[
        AutoEvalFramework, _AutoEvalTaskRunnerParams]:
        all_to_embed_per_res = {}
        all_to_embed_per_seq = {}
        framework_to_tuples = {}
        for framework_obj in self._frameworks_to_runners.keys():
            task_config_tuples, to_embed_per_res, to_embed_per_seq = get_unique_framework_sequences(
                framework=framework_obj,
                min_seq_length=self.min_seq_length,
                max_seq_length=self.max_seq_length,
                custom_storage_path=self.custom_storage_path,
                force_download=self.force_download,
                development_mode=self.development_mode)
            framework_to_tuples[framework_obj] = task_config_tuples
            all_to_embed_per_res.update(to_embed_per_res)
            all_to_embed_per_seq.update(to_embed_per_seq)

        embeddings_file_per_residue = None
        embeddings_file_per_sequence = None
        if len(all_to_embed_per_res) > 0 or len(all_to_embed_per_seq) > 0:
            embeddings_file_per_residue, embeddings_file_per_sequence = self._pre_embed(all_to_embed_per_res,
                                                                                        all_to_embed_per_seq,
                                                                                        devices=devices)

        framework_to_params = {}
        for i, (framework_obj, task_config_tuples) in enumerate(framework_to_tuples.items()):
            device = get_device(devices[i % len(devices)]) if devices else None
            runner_params = _AutoEvalTaskRunnerParams(task_config_tuples=task_config_tuples,
                                                      embeddings_file_per_residue=embeddings_file_per_residue,
                                                      embeddings_file_per_sequence=embeddings_file_per_sequence,
                                                      device=device)
            framework_to_params[framework_obj] = runner_params

        return framework_to_params

    def _pre_embed(self, all_unique_per_residue, all_unique_per_sequence,
                   devices: Optional[List[Union[str, torch.device]]] = None):
        embedder = setup_embedder(embedder_name=self.embedder_name,
                                  output_dir=self.output_dir,
                                  precomputed_per_residue_embeddings=self.precomputed_per_residue_embeddings,
                                  precomputed_per_sequence_embeddings=self.precomputed_per_sequence_embeddings,
                                  custom_embedder=self.custom_embedder,
                                  device=devices[0] if devices else None
                                  )
        # Embed
        print(f"Embedding {len(all_unique_per_residue)} sequences per_residue")
        embeddings_file_per_residue = embedder.per_residue_path(
            [seq_record.seq for _, seq_record in all_unique_per_residue.items()]
        )

        print(f"Embedding {len(all_unique_per_sequence)} sequences per_sequence")
        embeddings_file_per_sequence = embedder.per_sequence_path(
            [seq_record.seq for _, seq_record in all_unique_per_sequence.items()]
        )

        check_h5_file(name="per-residue", h5_path=embeddings_file_per_residue,
                      expected_length=len(all_unique_per_residue))
        check_h5_file(name="per-sequence", h5_path=embeddings_file_per_sequence,
                      expected_length=len(all_unique_per_sequence))

        print("Calculated embeddings successfully!")
        return embeddings_file_per_residue, embeddings_file_per_sequence

    def _general_task_setup(self, available_framework: AvailableFramework,
                            zero_shot_method: Optional[ZeroShotMethod] = None) -> Optional:
        framework_obj: AutoEvalFramework = validate_input(available_framework,
                                                          zero_shot_method=zero_shot_method,
                                                          min_seq_length=self.min_seq_length,
                                                          max_seq_length=self.max_seq_length)

        if framework_obj in self._frameworks_to_runners.keys():
            raise ValueError(f"Framework {framework_obj.get_name()} already added to AutoEval!")

        # Setup
        output_dir = setup_output_dir(base_dir=self.output_dir,
                                      embedder_name=self.embedder_name,
                                      framework_name=framework_obj.get_name())
        # Check if results already exist
        if not self.autoeval_report:
            self.autoeval_report = AutoEvalReport.loaded_or_empty(embedder_name=self.embedder_name,
                                                                  training_date=str(datetime.now().date().isoformat()),
                                                                  output_dir=self.output_dir)
        # Framework results already exist -> skip execution
        assert self.autoeval_report is not None
        maybe_framework_result = self.autoeval_report.maybe_framework_result(framework_name=framework_obj.get_name())
        if maybe_framework_result:
            dev_mode = maybe_framework_result.used_development_mode()
            full_evaluation_exists = not dev_mode  # Full evaluation always contains development mode so can be skipped
            full_evaluation_desired = not self.development_mode
            if full_evaluation_exists:
                print(f"Autoeval report for framework {available_framework} already exists, "
                      f"execution will be skipped!")
                self._results[framework_obj] = maybe_framework_result
            elif full_evaluation_desired:
                print(f"Autoeval report for framework exists, but only in development mode. "
                      f"Running full evaluation now!")

        return framework_obj, maybe_framework_result, output_dir

    def _supervised_task(self,
                         available_framework: AvailableFramework,
                         custom_output_observers: List[BiotrainerOutputObserver] = None, ):
        framework_obj, maybe_framework_result, output_dir = self._general_task_setup(available_framework)
        if maybe_framework_result:
            return self

        def runner_function(runner_params: _AutoEvalTaskRunnerParams):
            # Supervised tasks ignore the development_mode, because it can simply be switched on/off in the dashboard
            # (Looking at validation vs. test set performance)
            return autoeval_supervised_pipeline(
                embedder_name=self.embedder_name,
                framework=framework_obj,
                embeddings_file_per_residue=runner_params.embeddings_file_per_residue,
                embeddings_file_per_sequence=runner_params.embeddings_file_per_sequence,
                output_dir=output_dir,
                task_config_tuples=runner_params.task_config_tuples,
                min_seq_length=self.min_seq_length,
                max_seq_length=self.max_seq_length,
                custom_output_observers=custom_output_observers,
                device=runner_params.device)

        self._frameworks_to_runners[framework_obj] = _AutoEvalTaskRunner(framework=framework_obj,
                                                                         runner=runner_function)
        return self

    def pbc_supervised(self,
                       custom_output_observers: List[BiotrainerOutputObserver] = None,
                       ) -> AutoEval:
        """
        Add PBC supervised evaluation tasks to the AutoEval pipeline.

        :param custom_output_observers: Optional list of custom training output observers 
            for monitoring and logging training progress.
        :param dataset_subset: Optional list of dataset names to execute. If None, all datasets will be executed.
        :return: The AutoEval instance for method chaining.
        """
        return self._supervised_task(AvailableFramework.PBC_SUPERVISED, custom_output_observers)

    def flip(self,
             custom_output_observers: List[BiotrainerOutputObserver] = None,
             ) -> AutoEval:
        """
        Add FLIP supervised evaluation tasks to the AutoEval pipeline.

        :param custom_output_observers: Optional list of custom training output observers 
            for monitoring and logging training progress.
        :return: The AutoEval instance for method chaining.
        """
        return self._supervised_task(AvailableFramework.FLIP, custom_output_observers)

    def pgym(self, zero_shot_method: ZeroShotMethod):
        """
        Add PGYM zero-shot evaluation tasks to the AutoEval pipeline.

        :param zero_shot_method: The zero-shot method to use for evaluation 
            (e.g., ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS).
        :return: The AutoEval instance for method chaining.
        """
        framework_obj, maybe_framework_result, output_dir = self._general_task_setup(AvailableFramework.PGYM,
                                                                                     zero_shot_method=zero_shot_method)
        if maybe_framework_result:
            return self

        def runner_function(runner_params: _AutoEvalTaskRunnerParams):
            bioengineer = self.bioengineer
            if not bioengineer:
                bioengineer = BioEngineer.from_name(name=self.embedder_name, device=runner_params.device)
            return autoeval_zeroshot_pipeline(
                framework=framework_obj,
                embedder_name=self.embedder_name,
                autoeval_tasks=runner_params.task_config_tuples,
                zero_shot_method=zero_shot_method,
                output_dir=output_dir,
                bioengineer=bioengineer,
                device=runner_params.device,
            )

        self._frameworks_to_runners[framework_obj] = _AutoEvalTaskRunner(framework=framework_obj,
                                                                         runner=runner_function
                                                                         )
        return self

    def pbc_zeroshot_contact(self, zero_shot_method: ZeroShotMethod = ZeroShotMethod.JACOBIAN_CONTACT):
        """
        Add PBC zero-shot contact prediction evaluation tasks to the AutoEval pipeline.

        :param zero_shot_method: The zero-shot method to use for contact prediction. 
            Defaults to ZeroShotMethod.JACOBIAN_CONTACT.
        :return: The AutoEval instance for method chaining.
        """
        framework_obj, maybe_framework_result, output_dir = self._general_task_setup(
            AvailableFramework.PBC_ZEROSHOT_CONTACT,
            zero_shot_method=zero_shot_method)
        if maybe_framework_result:
            return self

        def runner_function(runner_params: _AutoEvalTaskRunnerParams):
            bioengineer = self.bioengineer
            if not bioengineer:
                bioengineer = BioEngineer.from_name(name=self.embedder_name, device=runner_params.device)
            return autoeval_zeroshot_contact_pipeline(
                framework=framework_obj,
                embedder_name=self.embedder_name,
                zero_shot_method=zero_shot_method,
                autoeval_tasks=runner_params.task_config_tuples,
                output_dir=output_dir,
                bioengineer=bioengineer,
                device=runner_params.device,
                development_mode=self.development_mode,
            )

        self._frameworks_to_runners[framework_obj] = _AutoEvalTaskRunner(framework=framework_obj,
                                                                         runner=runner_function
                                                                         )

        return self

    def pbc_supervised_contact(self):
        """
        Add PBC supervised contact prediction evaluation tasks to the AutoEval pipeline.

        :return: The AutoEval instance for method chaining.
        """
        framework_obj, maybe_framework_result, output_dir = self._general_task_setup(
            AvailableFramework.PBC_SUPERVISED_CONTACT)
        if maybe_framework_result:
            return self
        self._frameworks_to_runners[framework_obj] = _AutoEvalTaskRunner(framework=framework_obj, runner=
        lambda task_params: autoeval_supervised_contact_pipeline(
            framework=framework_obj,
            embedder_name=self.embedder_name,
            autoeval_tasks=task_params.task_config_tuples,
            output_dir=output_dir,
            device=task_params.device,
            custom_embedder=self.custom_embedder,
            development_mode=self.development_mode)
                                                                         )
        return self

    def run(self, device: Optional[Union[str, torch.device]] = None) -> AutoEvalReport:
        """
        Execute all configured evaluation tasks sequentially.

        :param device: Optional device specifier for embedding/model computations 
            (e.g., 'cuda:0', 'cuda:1', 'cpu'). If not specified, the default device is used.
        :return: An AutoEvalReport containing the evaluation results for all frameworks.
        :raises RuntimeError: If no progress or final report is returned from a task.
        :raises AssertionError: If the autoeval report is None before task execution.
        """
        framework_to_params = self._setup_runner_params(devices=[device] if device else None)

        # Make sure that autoeval report exists and all existing results are added and synced to disk
        assert self.autoeval_report is not None, f"Autoeval report is None before task execution!"
        for framework, existing_results in self._results.items():
            self.autoeval_report.add_framework_report(framework_name=framework.get_name(),
                                                      framework_mode=framework.get_mode(),
                                                      report=existing_results)
        self.autoeval_report.write(self.output_dir)

        # Execute tasks
        for framework, params in framework_to_params.items():
            task_runner = self._frameworks_to_runners[framework]
            assert task_runner is not None, f"Runner for framework {framework.get_name()} is None!"
            current_progress = None
            for progress in task_runner.runner(params):
                print(progress)
                current_progress = progress
            if current_progress is None:
                raise RuntimeError("No progress was returned from autoeval task!")
            final_report = current_progress.final_report
            if final_report is None:
                raise RuntimeError("No final report was returned from autoeval task!")
            self.autoeval_report.add_framework_report(framework_name=framework.get_name(),
                                                      framework_mode=framework.get_mode(),
                                                      report=final_report)
            self.autoeval_report.write(self.output_dir)

        return self.autoeval_report
