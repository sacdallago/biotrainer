import random
import h5py
import numpy as np
import torch
import time

from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
from biotrainer_core.data_classes import SequenceData, BootstrappedMetric
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalProgress, UnsupervisedFrameworkReport, \
    DEV_MODE_INDICATOR
from biotrainer_core.utils import str2bool
from biotrainer_core.input_files import read_FASTA
from biotrainer_core.functions.bootstrapping import get_mean_and_confidence_bounds
from typing import Optional, Dict, Tuple, List, Any, Generator

from ..core import AutoEvalFramework



@dataclass
class _EATPrediction:
    query_id: str
    query_label: str
    lookup_id: str
    lookup_label: str
    nn_d: float
    nn_iter: float


class _EATEvaluator:
    # Inspired by ProtTucker (EAT): https://github.com/Rostlab/EAT/blob/b82ea03688a894d3717019972a455b55b785cd53/eat.py#L45-L96

    def __init__(self, predictions: List[_EATPrediction]):
        self.ys, self.yhats, self.reliabilities = zip(
            *[(prediction.query_label, prediction.lookup_label, prediction.nn_d)
              for prediction in predictions
              if prediction.nn_iter == 0
              ]
            )

    def evaluate(self, set_name: str) -> List[BootstrappedMetric]:
        n_bootstrap_iterations = 1000
        confidence_level = 0.05

        n_total = len(self.ys)  # total number of predictions
        idx_list = range(n_total)

        ys = np.array(self.ys)  # ground truth
        yhats = np.array(self.yhats)  # lookup labels
        labels = sorted(set(ys) | set(yhats))

        # Using sklearn metrics instead of biotrainer metrics calculators because they can handle strings better
        acc = accuracy_score(ys, yhats)
        f1 = f1_score(ys, yhats, average="weighted", labels=labels)

        accs_btrap = []
        f1s_btrap = []
        for _ in range(n_bootstrap_iterations):
            rnd_subset = random.choices(idx_list, k=n_total)
            acc_bt = accuracy_score(ys[rnd_subset], yhats[rnd_subset])
            accs_btrap.append(acc_bt)
            f1_bt = f1_score(ys[rnd_subset], yhats[rnd_subset], average="weighted", labels=labels, zero_division=0)
            f1s_btrap.append(f1_bt)

        metric_dict = {"accuracy": (acc, accs_btrap),
                "macro-f1_score": (f1, f1s_btrap)}
        print(f"Metrics for {set_name}:")
        result = []
        for metric, (performance, bootstrapped_values) in metric_dict.items():

            mean, std, lower_bound, upper_bound = get_mean_and_confidence_bounds(np.array(bootstrapped_values),
                                                                                 dimension=0,
                                                                                 confidence_level=confidence_level)

            bt_metric = BootstrappedMetric(name=metric,
                                           mean=mean.item(),
                                           lower=lower_bound.item(),
                                           upper=upper_bound.item(),
                                           iterations=n_bootstrap_iterations,
                                           sample_size=n_total,
                                           confidence_level=confidence_level)
            result.append(bt_metric)
            print(f"\t - {bt_metric}")
        return result


def _pairwise_distance(lookup: torch.Tensor,
                       queries: torch.Tensor,
                       norm: int = 2,
                       use_double: bool = False):
    # Insipred by ProtTucker (EAT): https://github.com/Rostlab/EAT/blob/b82ea03688a894d3717019972a455b55b785cd53/eat.py#L338
    lookup = lookup.unsqueeze(dim=0)
    queries = queries.unsqueeze(dim=0)
    # double precision improves performance slightly but can be removed for speedy predictions (no significant difference in performance)
    if use_double:
        lookup = lookup.double()
        queries = queries.double()

    try:  # try to batch-compute pairwise-distance on GPU
        pdist = torch.cdist(lookup, queries, p=norm).squeeze(dim=0)
    except RuntimeError as e:
        print("Encountered RuntimeError: {}".format(e))
        print("Trying single query inference on GPU.")
        try:  # if OOM for batch-GPU, re-try single query pdist computation on GPU
            pdist = torch.stack(
                [torch.cdist(lookup, queries[0:1, q_idx], p=norm).squeeze(dim=0)
                 for q_idx in range(queries.shape[1])
                 ]
            ).squeeze(dim=-1).T

        except RuntimeError as e:  # if OOM for single GPU, re-try single query on CPU
            print("Encountered RuntimeError: {}".format(e))
            print("Trying to move single query computation to CPU.")
            lookup = lookup.to("cpu")
            queries = queries.to("cpu")
            pdist = torch.stack(
                [torch.cdist(lookup, queries[0:1, q_idx], p=norm).squeeze(dim=0)
                 for q_idx in range(queries.shape[1])
                 ]
            ).squeeze(dim=-1).T

    lookup_pdist = pdist.shape[0]
    query_pdist = pdist.shape[1]

    print(f"Calculated pairwise distance matrix [{lookup_pdist},{query_pdist}] "
          f"for {lookup_pdist} lookup embeddings and {query_pdist} query embeddings!")
    return pdist


def _get_nearest_neighbours(
        lookup_embeddings,
        query_embeddings,
        lookup_seqs: List[SequenceData],
        query_seqs: List[SequenceData],
        config: Dict[str, Any],
        random: bool = False):
    # Inspired by ProtTucker (EAT): https://github.com/Rostlab/EAT/blob/b82ea03688a894d3717019972a455b55b785cd53/eat.py#L372
    start = time.time()
    num_nn = config["num_nn"]
    threshold = config["threshold"]  # Euclidean distance threshold

    p_dist = _pairwise_distance(lookup_embeddings, query_embeddings)

    if random:  # this is only needed for benchmarking against random background
        print("Doing random predictions!")
        nn_dists, nn_idxs = torch.topk(torch.rand_like(p_dist), num_nn, largest=False, dim=0)
    else:  # infer nearest neighbor indices
        nn_dists, nn_idxs = torch.topk(p_dist, num_nn, largest=False, dim=0)

    print("Computing NN took: {:.4f}[s]".format(time.time() - start))

    nn_dists = nn_dists.to("cpu")
    nn_idxs = nn_idxs.to("cpu")
    predictions = []
    for idx, query_seq in enumerate(query_seqs):
        nn_idx = nn_idxs[:, idx]
        nn_dist = nn_dists[:, idx]
        for nn_iter, (nn_index, nn_distance) in enumerate(zip(nn_idx, nn_dist)):
            # index of nearest neighbour (nn) in train set
            nn_index, nn_distance = int(nn_index), float(nn_distance)
            # if a threshold is passed, skip all proteins above this threshold
            if threshold is not None and nn_distance > threshold:
                continue
            # get id of nn (infer annotation)
            lookup_seq = lookup_seqs[nn_index]
            lookup_label = lookup_seq.label
            query_label = query_seq.label
            prediction = _EATPrediction(query_id=query_seq.seq_id,
                                        query_label=str(query_label),
                                        lookup_id=lookup_seq.seq_id,
                                        lookup_label=str(lookup_label),
                                        nn_d=nn_distance,
                                        nn_iter=nn_iter)
            predictions.append(prediction)
    end = time.time()
    print("Assigning NN predictions took: {:.4f}[s]".format(end - start))
    return predictions


def eat(seq_records: List[SequenceData],
        embeddings_file_per_sequence: Path,
        config: Dict[str, Any]):
    lookup_seqs = {seq_record.get_hash(): seq_record for seq_record in seq_records if seq_record.set == "lookup"}
    test_seqs = {seq_record.get_hash(): seq_record for seq_record in seq_records if seq_record.set == "test"}

    print("Loading embeddings..")
    with h5py.File(embeddings_file_per_sequence, "r") as embd_file:
        lookup_embeddings = torch.stack([torch.tensor(embd_file[seq_hash]) for seq_hash in lookup_seqs.keys()])
        test_embeddings = torch.stack([torch.tensor(embd_file[seq_hash]) for seq_hash in test_seqs.keys()])

    print("Calculating lookup predictions..")
    test_predictions = _get_nearest_neighbours(lookup_embeddings=lookup_embeddings,
                                               query_embeddings=test_embeddings,
                                               lookup_seqs=list(lookup_seqs.values()),
                                               query_seqs=list(test_seqs.values()),
                                               config=config,
                                               random=False)
    random_test_predictions = _get_nearest_neighbours(lookup_embeddings=lookup_embeddings,
                                                      query_embeddings=test_embeddings,
                                                      lookup_seqs=list(lookup_seqs.values()),
                                                      query_seqs=list(test_seqs.values()),
                                                      config=config,
                                                      random=True)
    print("Evaluating predictions..")
    test_evaluator = _EATEvaluator(predictions=test_predictions)
    test_results = test_evaluator.evaluate(set_name="test")
    test_random_evaluator = _EATEvaluator(predictions=random_test_predictions)
    test_random_results = test_random_evaluator.evaluate(set_name="test_random")
    return {"test": test_results,
            "test_random": test_random_results}


def autoeval_unsupervised_pipeline(embedder_name: str,
                                   framework: AutoEvalFramework,
                                   embeddings_file_per_sequence: Optional[Path],
                                   task_config_tuples: List[Tuple[AutoEvalTask, Dict[str, Any]]],
                                   output_dir: Path,
                                   min_seq_length: int,
                                   max_seq_length: int,
                                   development_mode: bool,
                                   device=None,
                                   ) -> Generator[AutoEvalProgress, None, None]:
    assert embeddings_file_per_sequence is not None, f"Missing embeddings file for unsupervised pipeline!"

    # Framework results do not exist yet -> execute biotrainer
    unsupervised_framework_report = UnsupervisedFrameworkReport.empty(min_seq_len=min_seq_length,
                                                                      max_seq_len=max_seq_length)
    task_names = [task.combined_name() + (DEV_MODE_INDICATOR if development_mode else "")
                  for task, _ in task_config_tuples]
    print(f"The following tasks will be executed in order: {task_names} (total {len(task_names)})")
    completed_tasks = 0
    total_tasks = len(task_config_tuples)
    current_task_name = ""
    for task, config in task_config_tuples:
        current_task_name = task.combined_name() + (DEV_MODE_INDICATOR if development_mode else "")
        print(f"Running task {current_task_name}...")
        yield AutoEvalProgress(completed_tasks=completed_tasks, total_tasks=total_tasks,
                               current_task_name=current_task_name,
                               current_framework_name=framework.get_name())

        fasta_file = task.input_files[0]
        seq_records = read_FASTA(fasta_file)
        assert len(seq_records) > 0, f"No sequences found in {fasta_file}!"

        if development_mode:
            dev_mode_subsample = [seq_record for seq_record in seq_records
                                  if str2bool(seq_record.get_attribute("DEV_MODE") or "False")]
            assert 0 < len(dev_mode_subsample) < len(seq_records), f"Development mode subsample size incorrect!"
            seq_records = dev_mode_subsample

        result = eat(seq_records=seq_records,
                     embeddings_file_per_sequence=embeddings_file_per_sequence,
                     config=config)

        unsupervised_framework_report.update_result(combined_task_name=current_task_name, result=result)

        completed_tasks += 1
        print(f"Finished task execution for {current_task_name}!")

    print(f"Autoeval supervised pipeline on framework {framework.get_name()} "
          f"for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=unsupervised_framework_report)
