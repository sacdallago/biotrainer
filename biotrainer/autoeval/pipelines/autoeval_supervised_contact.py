"""
This module contains the implementation of the AutoEval pipeline for supervised contact prediction.
Heavily inspired by: https://github.com/chandar-lab/AMPLIFY/blob/main/examples/contact_prediction.ipynb
"""

from __future__ import annotations

import torch
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from typing import Optional, List, Dict, Any, Tuple

from biotrainer_core.data_classes import SequenceData, ContactDatasetResult
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalProgress, ContactFrameworkReport

from biotrainer_core.input_files import load_contact_map, read_FASTA

from .autoeval_pipeline_utils import subsample_seq_records_for_contact_development_mode

from ..core import AutoEvalFramework

from ...shared import get_device
from ...embedding import get_embedding_service
from ...shared.metrics import evaluate_contact_dataset
from ...embedding.huggingface import HuggingfaceTransformerEmbedder


@dataclass
class LogisticRegressionHyperParameters:
    solver: str = "liblinear"
    l1_ratio: float = 1
    C: float = 1  # Permutated
    seed: int = 0

    @staticmethod
    def all() -> List[LogisticRegressionHyperParameters]:
        hps = []
        for c in [0.5, 1, 1.5]:
            hps.append(LogisticRegressionHyperParameters(C=c))
        return hps


@dataclass
class _PerProteinData:
    seq_id: str
    sequence: str
    ground_truth_contact_map: np.ndarray
    attention_map: Optional[np.ndarray] = None


@dataclass
class _InputDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    val_dataset: List[_PerProteinData]
    test_datasets: Dict[str, List[_PerProteinData]]  # [(seq_id, attention_map, ground_truth_contact_map)]


def _generate_flat_pairwise_dataset_input(seq_data: List[SequenceData], embedding_service, contacts_dir_path):
    min_sep = 6  # https://github.com/chandar-lab/AMPLIFY/blob/main/examples/contact_prediction.ipynb

    x = []
    y = []
    for record in seq_data:
        seq_id = record.seq_id
        sequence = record.seq
        ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
        ground_truth_contact_map = load_contact_map(path=ground_truth_contact_map_path,
                                                    sequence=sequence,
                                                    structure_id=seq_id)

        pos = np.arange(ground_truth_contact_map.shape[0])
        diag_idx = np.expand_dims(pos, axis=0) - np.expand_dims(pos, axis=1) >= min_sep

        attention_map = embedding_service._embedder.compute_attention_map(sequence=sequence)

        x.extend(attention_map[diag_idx, :].reshape(-1, attention_map.shape[-1]).to(torch.float32))
        y.extend(ground_truth_contact_map[diag_idx].reshape(-1))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return x, y


def _generate_per_protein_dataset_input(seq_data: List[SequenceData],
                                        contacts_dir_path,
                                        embedding_service: Optional = None) -> List[_PerProteinData]:
    """ Generate per protein dataset input (val/test). If embedding_service is None (test),
        attention maps need to be computed lazily later """
    protein_inputs = []

    for record in seq_data:
        seq_id = record.seq_id
        sequence = record.seq
        ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
        ground_truth_contact_map = load_contact_map(path=ground_truth_contact_map_path,
                                                    sequence=sequence,
                                                    structure_id=seq_id)
        attention_map = None
        if embedding_service is not None:
            attention_map = embedding_service._embedder.compute_attention_map(sequence=sequence)

        per_protein_data = _PerProteinData(seq_id=seq_id,
                                           sequence=sequence,
                                           attention_map=attention_map,
                                           ground_truth_contact_map=ground_truth_contact_map,
                                           )
        protein_inputs.append(per_protein_data)

    return protein_inputs


def _load_data_and_generate_attention_maps(dataset_map: dict, embedding_service,
                                           development_mode: bool) -> Tuple[_InputDataset, List[str]]:
    # Train
    dataset_dir_path = dataset_map["train"]
    fasta_file_path = dataset_dir_path / "extracted_sequences.fasta"
    train_seqs = read_FASTA(fasta_file_path)

    contacts_dir_path = dataset_dir_path / "contacts"
    x_train, y_train = _generate_flat_pairwise_dataset_input(train_seqs, embedding_service, contacts_dir_path)

    # Val
    dataset_dir_path = dataset_map["val"]
    fasta_file_path = dataset_dir_path / "extracted_sequences.fasta"
    val_seqs = read_FASTA(fasta_file_path)

    contacts_dir_path = dataset_dir_path / "contacts"
    val_dataset = _generate_per_protein_dataset_input(val_seqs, contacts_dir_path, embedding_service=embedding_service)

    # Test
    test_datasets = {}
    development_ids = []
    for test_path in dataset_map["test"]:
        test_name = test_path.stem
        test_fasta_file_path = test_path / "extracted_sequences.fasta"
        test_seqs = read_FASTA(test_fasta_file_path)
        subsampled_test_seqs = subsample_seq_records_for_contact_development_mode(test_seqs)
        development_ids.extend([seq_record.seq_id for seq_record in subsampled_test_seqs])
        if development_mode:
            test_seqs = subsampled_test_seqs

        contacts_dir_path = test_path / "contacts"
        # Lazy compute attention maps later
        test_datasets[test_name] = _generate_per_protein_dataset_input(test_seqs, contacts_dir_path,
                                                                       embedding_service=None)

    return (_InputDataset(x_train=x_train, y_train=y_train, val_dataset=val_dataset, test_datasets=test_datasets),
            development_ids)


def _train_logistic_regression(current_task_name: str, input_dataset: _InputDataset) -> LogisticRegression:
    x_train = input_dataset.x_train
    y_train = input_dataset.y_train
    val_dataset = input_dataset.val_dataset

    hyper_params = LogisticRegressionHyperParameters.all()
    best_clf = None
    best_long_p_at_l_val = -np.inf
    for idx, hp in enumerate(hyper_params):
        print(f"{current_task_name}: Training classifier {idx} (Total: {len(hyper_params)})")
        # Train classifier on training data
        # Logistic Regression (careful, liblinear does not support int64!)
        clf = LogisticRegression(solver=hp.solver, l1_ratio=hp.l1_ratio, C=hp.C)
        clf.fit(x_train, y_train)

        # Test on Val dataset
        val_dataset_result = _test_logistic_regression(clf=clf, test_set_name=f"Val-Clf{idx}",
                                                       per_protein_data=val_dataset,
                                                       )
        long_p_at_l2_val = val_dataset_result.long_PatL2()
        print(f"{current_task_name}: long_P@L2 for hyperparameters {hp}: {long_p_at_l2_val}")
        if long_p_at_l2_val is None:
            raise ValueError(f"long_P@L2 not found in Val dataset result for hyperparameters: {hp}")
        if long_p_at_l2_val > best_long_p_at_l_val:
            best_clf = clf
            best_long_p_at_l_val = long_p_at_l2_val
    print(f"{current_task_name}: After {len(hyper_params)} hyper param combinations, "
          f"found best long_P@L2: {best_long_p_at_l_val} for linear classifier!")
    assert best_clf is not None, "Best classifier not found!"
    return best_clf


def _test_logistic_regression(clf: LogisticRegression, test_set_name: str,
                              per_protein_data: List[_PerProteinData],
                              embedding_service: Optional = None) -> ContactDatasetResult:
    def predict_function(data_point: _PerProteinData):
        if data_point.attention_map is not None:
            attention_map = data_point.attention_map
        else:
            assert embedding_service is not None, "Attention map not provided, but embedding service is None!"
            attention_map = embedding_service._embedder.compute_attention_map(sequence=data_point.sequence)

        return (clf.predict_proba(attention_map.reshape(-1, attention_map.shape[-1]))[:, 1]
                .reshape(data_point.ground_truth_contact_map.shape))

    def evaluate():
        yield from evaluate_contact_dataset(dataset_name=test_set_name,
                                            items=per_protein_data,
                                            predict_func=predict_function,
                                            get_ground_truth_func=lambda d: d.ground_truth_contact_map,
                                            get_seq_id_func=lambda d: d.seq_id
                                            )

    for maybe_single_result, maybe_dataset_result in evaluate():
        # No caching for supervised task, so only check for dataset result
        if maybe_dataset_result is not None:
            return maybe_dataset_result

    assert False, "No dataset result returned!"


def autoeval_supervised_contact_pipeline(framework: AutoEvalFramework,
                                         embedder_name: str,
                                         output_dir: Path,
                                         autoeval_tasks: List[Tuple[AutoEvalTask, Dict[str, Any]]],
                                         development_mode: bool,
                                         device=None):
    embedding_service = get_embedding_service(embedder_name=embedder_name, device=get_device(device),
                                              custom_tokenizer_config=None)
    if not isinstance(embedding_service._embedder, HuggingfaceTransformerEmbedder):
        raise ValueError(f"Only HuggingfaceTransformers are supported for supervised contact tasks, "
                         f"but got {embedding_service._embedder}!")

    autoeval_tasks = [task for task, _ in autoeval_tasks]  # Ignore config for supervised contact
    assert len(autoeval_tasks) == 1, "Only one supervised contact task is supported!"
    task = autoeval_tasks[0]

    task_names = [task.combined_name() for task in autoeval_tasks]
    print(f"The following tasks will be executed in order: {task_names} (total {len(task_names)})")
    total_tasks = len(task_names)
    current_task_name = task.combined_name()
    print(f"Running task {current_task_name}...")
    yield AutoEvalProgress(completed_tasks=0,
                           total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name())

    # (1) Set up dataset map from input files
    print(f"{current_task_name}: Loading data..")
    dataset_map = {"train": None,
                   "val": None,
                   "test": []
                   }
    input_dirs = task.input_files
    for input_dir in input_dirs:
        dir_name = input_dir.name
        if dir_name in dataset_map:
            dataset_map[dir_name] = input_dir
        else:
            dataset_map["test"].append(input_dir)

    assert dataset_map["train"] is not None, f"Missing train dataset for task: {current_task_name}"
    assert dataset_map["val"] is not None, f"Missing val dataset for task: {current_task_name}"
    assert len(dataset_map["test"] or []) > 0, f"Missing test datasets for task: {current_task_name}"

    # (2) Data Collection
    input_dataset, development_ids = _load_data_and_generate_attention_maps(dataset_map=dataset_map,
                                                           embedding_service=embedding_service,
                                                           development_mode=development_mode)
    supervised_contact_framework_report = ContactFrameworkReport.empty(development_ids=development_ids)

    # (3) Training
    print(f"{current_task_name}: Training classifiers..")
    best_clf = _train_logistic_regression(current_task_name=current_task_name, input_dataset=input_dataset)

    # (4) Test
    test_datasets = input_dataset.test_datasets
    for test_set_name, test_data in test_datasets.items():
        dataset_result = _test_logistic_regression(clf=best_clf, test_set_name=test_set_name,
                                                   per_protein_data=test_data,
                                                   embedding_service=embedding_service)
        supervised_contact_framework_report.update_result(task_name=test_set_name,
                                                          dataset_result=dataset_result)
    print(f"Finished task {current_task_name}!")

    print(f"Autoeval supervised contact pipeline on framework {framework.get_name()} "
          f"for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=supervised_contact_framework_report)
