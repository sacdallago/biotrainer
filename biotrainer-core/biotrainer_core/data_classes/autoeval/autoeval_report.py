from __future__ import annotations

import hashlib
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Union, Optional, List, Tuple

from .autoeval_mode import AutoEvalMode
from .autoeval_task import AutoEvalTask
from .autoeval_flip_datasets import all_flip_datasets
from .autoeval_pbc_datasets import all_pbc_supervised_datasets

from ..embedding_stats import EmbeddingStats
from ..biotrainer_model_result import BiotrainerModelResult
from ..contact import ContactDatasetResult, ContactSingleProteinResult
from ..bioengineer_data_classes import ZeroShotMethod, RankingResult, AggregatedRankingResult


def _aggregate_dfs(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        return pd.concat(dfs, ignore_index=True)
    except ValueError:
        return None


def _maybe_metric_abs(metric_name: str, mean: float, lower: float, upper: float) -> Tuple[float, float, float]:
    """ Convert metric to absolute for comparison (correlation coefficients)"""
    m = (metric_name or "").lower()
    names_for_abs = ["mcc", "spearman", "scc", "spearmans-corr-coeff"]
    if m in names_for_abs:
        try:
            if mean < 0:
                upper = abs(float(lower))
                lower = abs(float(upper))
            else:
                lower = abs(float(lower))
                upper = abs(float(upper))
            mean = abs(float(mean))
            return mean, lower, upper
        except Exception:
            return mean, lower, upper
    return mean, lower, upper


class FrameworkReport(ABC, BaseModel):
    @abstractmethod
    def summary(self, development_mode: bool = False):
        raise NotImplementedError

    @abstractmethod
    def number_tasks(self):
        raise NotImplementedError

    @abstractmethod
    def get_task_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def to_df(self, all_metrics: bool, development_mode: bool = False) -> pd.DataFrame:
        """ Convert to pandas dataframe."""
        raise NotImplementedError

    def used_development_mode(self) -> bool:
        """ Whether development mode was used in the autoeval pipeline"""
        return False


class SupervisedFrameworkReport(FrameworkReport):
    min_seq_len: Optional[int] = Field(default=None, description="Minimum sequence length used during evaluation")
    max_seq_len: Optional[int] = Field(default=None, description="Maximum sequence length used during evaluation")
    results: Dict[str, BiotrainerModelResult] = Field(description="Supervised autoeval results")

    @classmethod
    def empty(cls, min_seq_len: Optional[int], max_seq_len: Optional[int]) -> SupervisedFrameworkReport:
        return cls(min_seq_len=min_seq_len, max_seq_len=max_seq_len, results={})

    def update_result(self, combined_task_name: str, result: BiotrainerModelResult):
        self.results[combined_task_name] = result

    @staticmethod
    def maybe_load_existing_result(embedder_name: str, task_output_dir: Path):
        task_out_file_path = task_output_dir / "out.yml"
        if not task_out_file_path.exists():
            return None
        try:
            task_output = BiotrainerModelResult.from_file(task_out_file_path)
            if task_output.config["embedder_name"] == embedder_name and len(task_output.test_results) > 0:
                return task_output
            return None  # File does not seem to be valid
        except Exception:
            return None

    def accumulated_embedding_stats(self) -> Optional[EmbeddingStats]:
        embedding_stats = None
        for result in self.results.values():
            result_stats = EmbeddingStats.from_biotrainer_result(result)
            if embedding_stats is None:
                embedding_stats = result_stats
            else:
                embedding_stats = embedding_stats.accumulate_results(result_stats)
        return embedding_stats

    def summary(self, development_mode: bool = False):
        print(f"(Minimum sequence length: {self.min_seq_len}, Maximum sequence length: {self.max_seq_len})")
        task_names = self.results.keys()
        print(f"Total tasks: {len(task_names)}")
        print("Results:")

        for task_name in task_names:
            metrics = self.extract_metrics(task_name, development_mode=development_mode, all_metrics=False)
            for metric in metrics:
                print(
                    f"{metric['task_name']} ({metric['protocol']}) - {metric['test_set_name']} - "
                    f"{metric['evaluation_metric']}: {metric['mean']} ({metric['lower']} - {metric['upper']})"
                )

    def extract_metrics(self, combined_task_name: str, development_mode: bool = False,
                        all_metrics: bool = False) -> list[dict]:
        """Extract metrics for a given task."""
        framework_to_datasets = {"PBC_SUPERVISED": all_pbc_supervised_datasets(), "FLIP": all_flip_datasets()}

        metrics = []
        try:
            framework_name, dataset_name, split_name = AutoEvalTask.split_combined_name(combined_task_name)
            datasets = framework_to_datasets[framework_name.upper()]
            evaluation_metric = datasets[dataset_name].evaluation_metric if not all_metrics else None
            protocol = datasets[dataset_name].protocol.name
            if development_mode:
                metrics.extend(self._extract_metrics_val_set(combined_task_name, evaluation_metric, protocol))
            else:
                metrics.extend(self._extract_metrics_test_set(combined_task_name, evaluation_metric, protocol))
        except KeyError:
            print(f"Warning: Task {combined_task_name} not found.")
        return metrics

    def _extract_metrics_val_set(self,
                                 combined_task_name: str,
                                 evaluation_metric: Optional[str],
                                 protocol: str) -> list[dict]:
        val_results = self.results[combined_task_name].training_results["hold_out"].best_epoch_metrics.validation
        metric_values = {evaluation_metric: val_results[evaluation_metric]} if evaluation_metric else val_results

        metrics = []
        for metric_name, metric_value in metric_values.items():
            # TODO Bootstrapping for validation set on best training result
            metric_mean = round(metric_value, 3)
            metric_lower = round(metric_value, 3)
            metric_upper = round(metric_value, 3)

            metrics.append({
                "task_name": combined_task_name,
                "protocol": protocol,
                "test_set_name": "validation",
                "evaluation_metric": metric_name,
                "mean": metric_mean,
                "lower": metric_lower,
                "upper": metric_upper
            })
        return metrics

    def _extract_metrics_test_set(self,
                                  combined_task_name: str,
                                  evaluation_metric: Optional[str],
                                  protocol: str):
        test_results = self.results[combined_task_name].test_results

        metrics = []
        for test_set_name, test_set_result in test_results.items():
            bootstrapping = test_set_result.bootstrapped_metrics or []
            bootstrapping = {b_res.name: b_res for b_res in bootstrapping}
            bootstrapping = {
                evaluation_metric: bootstrapping[evaluation_metric]} if evaluation_metric else bootstrapping
            for metric_name, metric_value in bootstrapping.items():
                metric_mean = round(metric_value.mean, 3)
                metric_lower = round(metric_value.lower, 3)
                metric_upper = round(metric_value.upper, 3)

                metrics.append({
                    "task_name": combined_task_name,
                    "protocol": protocol,
                    "test_set_name": test_set_name,
                    "evaluation_metric": metric_name,
                    "mean": metric_mean,
                    "lower": metric_lower,
                    "upper": metric_upper
                })
        return metrics

    def to_df(self, all_metrics: bool, development_mode: bool = False) -> pd.DataFrame:
        rows = []

        for task in self.get_task_names():
            framework_name, dataset_name, _ = AutoEvalTask.split_combined_name(task)
            for m in self.extract_metrics(task, development_mode=development_mode, all_metrics=all_metrics):
                # Label like: Task\n(TestSet - Metric) if test set != 'test' else Task\n(Metric)
                test_set = m["test_set_name"]
                metric_name = m["evaluation_metric"]
                if test_set != "test":
                    label = f"{dataset_name}\n({test_set} - {metric_name})"
                else:
                    label = f"{dataset_name}\n({metric_name})"
                mean, lower, upper = _maybe_metric_abs(metric_name,
                                                       mean=m["mean"], lower=m["lower"], upper=m["upper"])
                rows.append({
                    "TaskLabel": label,
                    "Task": task,
                    'Protocol': m['protocol'],
                    "Test Set": test_set,
                    "Metric": metric_name,
                    "Mean": mean,
                    "Lower": lower,
                    "Upper": upper
                })
        df = pd.DataFrame(rows)
        return df

    def number_tasks(self):
        return len(self.results.keys())

    def get_task_names(self) -> List[str]:
        return list(self.results.keys())


class ZeroShotFrameworkReport(FrameworkReport):
    model_config = {"use_enum_values": True}

    method: ZeroShotMethod = Field(description="Scoring method used")
    task_results: Dict[str, AggregatedRankingResult] = Field(description="Aggregated ranking results "
                                                                         "(task_name -> AggregatedRankingResult)")
    task_results_dev: Dict[str, AggregatedRankingResult] = Field(description="Aggregated ranking results "
                                                                             "task results for development mode")
    task_members: Dict[str, List[str]] = Field(description="Datasets contributing to each aggregated task "
                                                           "(task_name -> [dataset_name]).")
    individual_results: Dict[str, RankingResult] = Field(description="Individual autoeval task results "
                                                                     "(dataset_name -> RankingResult)")
    development_ids: List[str] = Field(default=list, description="All protein ids that are used in development mode")

    @model_validator(mode='after')
    def check_method(self):
        if isinstance(self.method, str):
            self.method = ZeroShotMethod(self.method)
        return self

    @classmethod
    def empty(cls, method: ZeroShotMethod, development_ids: List[str]) -> ZeroShotFrameworkReport:
        return cls(method=method, task_results={}, task_results_dev={}, individual_results={},
                   task_members={}, development_ids=development_ids)

    def aggregate(self, task_name: str, individual_results: Dict[str, RankingResult]):
        self.individual_results.update(individual_results)
        self.task_members[task_name] = list(individual_results.keys())
        self.task_results[task_name] = AggregatedRankingResult.aggregate(list(individual_results.values()))
        individual_results_dev = {dataset_name: result for dataset_name, result in individual_results.items()
                                  if dataset_name in self.development_ids}
        self.task_results_dev[task_name] = AggregatedRankingResult.aggregate(list(individual_results_dev.values()))

    def summary(self, development_mode: bool = False):
        print(f"Zero-shot method: {self.method.value}")
        print(f"Total tasks: {len(self.task_results)}")
        print("Results:")
        for combined_task_name, result in self.task_results.items():
            print(f"{combined_task_name}: "
                  f"\t SCC:  {result.scc_score()}"
                  f"\t NDCG: {result.ndcg_score()}")

    def to_df(self, all_metrics: bool, development_mode: bool = False) -> pd.DataFrame:
        rows = []
        result_dict = self.task_results_dev if development_mode else self.task_results
        for task in self.get_task_names():
            framework_name, _, _ = AutoEvalTask.split_combined_name(task)
            ranking_result = result_dict.get(task)
            if ranking_result is None:
                continue
            all_zs_metrics = [ranking_result.scc,
                              ranking_result.ndcg]  # Only two metrics for zero shot so we always keep both
            for metric in all_zs_metrics:
                name = metric.name
                mean, lower, upper = _maybe_metric_abs(name,
                                                       mean=metric.mean, lower=metric.lower, upper=metric.upper)
                rows.append({
                    "TaskLabel": f"{task}\n({name})",
                    "Task": task,
                    "Metric": name,
                    "Mean": round(mean, 3),
                    "Lower": round(lower, 3),
                    "Upper": round(upper, 3),
                })
        rows = sorted(rows, key=lambda x: 'virus' in x['Task'], reverse=True)
        return pd.DataFrame(rows)

    def number_tasks(self):
        return len(self.task_results)

    def get_task_names(self) -> List[str]:
        return list(self.task_results.keys())

    def used_development_mode(self) -> bool:
        return len(self.individual_results) == len(self.development_ids)


class ZeroShotCachedResults(BaseModel):
    """ Utility class for storing cached results for zero-shot evaluation """
    embedder_name: str = Field(description="Name of the embedder")
    method: ZeroShotMethod = Field(description="Scoring method used")
    individual_results: Dict[str, RankingResult] = Field(description="Individual autoeval task results "
                                                                     "(dataset_name -> RankingResult)")

    @staticmethod
    def get_file_name(method: ZeroShotMethod):
        return f"zero_shot_cached_results_{method.value}.json"

    @classmethod
    def from_json_file(cls, file_path: Union[Path, str]) -> ZeroShotCachedResults:
        """Load ZeroShotCachedResults from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def empty(cls, embedder_name: str, method: ZeroShotMethod) -> ZeroShotCachedResults:
        return cls(embedder_name=embedder_name, method=method, individual_results={})

    @classmethod
    def loaded_or_empty(cls,
                        embedder_name: str,
                        method: ZeroShotMethod,
                        output_dir: Path) -> ZeroShotCachedResults:
        report_file_path = output_dir / cls.get_file_name(method)
        if report_file_path.exists():
            report = cls.from_json_file(report_file_path)
            assert report.embedder_name == embedder_name and report.method == method
            return report
        return cls.empty(embedder_name, method)

    def maybe_cached_result(self, dataset_name: str) -> Optional[RankingResult]:
        return self.individual_results.get(dataset_name, None)

    def update_and_sync(self, dataset_name: str, result: RankingResult, output_dir: Path):
        self.individual_results[dataset_name] = result
        self._write_to_file(output_dir=output_dir)

    def _write_to_file(self, output_dir: Union[Path, str]):
        file_path = output_dir / self.get_file_name(method=self.method)
        with open(file_path, 'w') as f:
            f.write(self.model_dump_json(indent=4))


class ContactFrameworkReport(FrameworkReport):
    model_config = {"use_enum_values": True}

    method: Optional[ZeroShotMethod] = Field(default=None,
                                             description="Contact method used. "
                                                         "Only applicable for zero-shot contact prediction")
    task_results: Dict[str, ContactDatasetResult] = Field(description="Results per tasks, i.e. per dataset (e.g. casp)")
    task_results_dev: Dict[str, ContactDatasetResult] = Field(description="Results per tasks for development mode, "
                                                                          "i.e. per dataset (e.g. casp)")
    task_members: Dict[str, List[str]] = Field(description="Datasets contributing to each "
                                                           "task (e.g. casp -> protein_id)")
    per_protein_results: Dict[str, ContactSingleProteinResult] = Field(
        description="Cached per protein results, stacking"
                    " up to the final dataset result (seq_id -> ContactSingleProteinResult)")
    development_ids: List[str] = Field(default=list, description="All protein ids that are used in development mode")

    @model_validator(mode='after')
    def check_method(self):
        if isinstance(self.method, str):
            self.method = ZeroShotMethod(self.method)
            if self.method != ZeroShotMethod.JACOBIAN_CONTACT:
                raise ValueError(f"Invalid contact method: {self.method}")
        return self

    @classmethod
    def empty(cls, method: Optional[ZeroShotMethod] = None) -> ContactFrameworkReport:
        return cls(method=method, task_results={}, task_results_dev={},
                   task_members={}, per_protein_results={}, development_ids=[])

    def update_result(self, task_name: str,
                      per_protein_results: Dict[str, ContactSingleProteinResult],
                      dataset_result: ContactDatasetResult,
                      dataset_result_dev: ContactDatasetResult):
        self.task_results[task_name] = dataset_result
        self.task_results_dev[task_name] = dataset_result_dev
        new_keys = set(per_protein_results.keys()) - set(self.per_protein_results.keys())
        self.task_members[task_name] = list(new_keys)
        self.per_protein_results.update(per_protein_results)

    def update_development_ids(self, development_ids: List[str]):
        self.development_ids.extend(development_ids)

    def summary(self, development_mode: bool = False):
        if self.method is not None:
            print(f"Zero-shot contact method: {self.method.value}")
        print(f"Total tasks: {len(self.task_results)}")
        print("Results:")
        for combined_task_name, result in self.task_results.items():
            print(f"{combined_task_name}: "
                  f"\t Results:  {result}")
            # TODO: add detailed print of metrics!!

    def to_df(self, all_metrics: bool, development_mode: bool = False) -> pd.DataFrame:
        rows = []
        primary_evaluation_metric = "long_P@L2"  # TODO Find better place for this constant
        for task, rr in self.task_results.items():
            if development_mode and task not in self.development_ids:
                continue
            task = task.split("-")[-1]
            if rr is None:
                continue
            contact_metrics = rr.aggregated_result
            contact_metrics = contact_metrics if all_metrics else [m for m in contact_metrics
                                                                   if m.name == primary_evaluation_metric]
            for metric in contact_metrics:
                name = metric.name
                mean, lower, upper = _maybe_metric_abs(name,
                                                       mean=metric.mean, lower=metric.lower,
                                                       upper=metric.upper)
                rows.append({
                    "TaskLabel": f"{task}\n({name})",
                    "Task": task,
                    "Metric": name,
                    "Mean": round(mean, 3),
                    "Lower": round(lower, 3),
                    "Upper": round(upper, 3),
                })
        return pd.DataFrame(rows)

    def number_tasks(self):
        return len(self.task_results)

    def get_task_names(self) -> List[str]:
        return [task_name.split("-")[-1] for task_name in self.task_results.keys()]

    def used_development_mode(self) -> bool:
        return len(self.task_results) == len(self.development_ids)


class ZeroShotContactCachedResults(BaseModel):
    """ Utility class for storing cached results for zero-shot contact evaluation """
    embedder_name: str = Field(description="Name of the embedder")
    method: ZeroShotMethod = Field(
        description="Contact method used")  # Note - only one applicable zeroshot contact method as of now!
    per_protein_results: Dict[str, ContactSingleProteinResult] = Field(
        description="Cached per protein results, stacking"
                    " up to the final dataset result (seq_id -> ContactSingleProteinResult)")

    @staticmethod
    def get_file_name(method: ZeroShotMethod):
        return f"zero_shot_contact_cached_results_{method.value}.json"

    @classmethod
    def from_json_file(cls, file_path: Union[Path, str]) -> ZeroShotContactCachedResults:
        """Load ZeroShotContactCachedResults from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def empty(cls, embedder_name: str, method: ZeroShotMethod) -> ZeroShotContactCachedResults:
        return cls(embedder_name=embedder_name, method=method, per_protein_results={})

    @classmethod
    def loaded_or_empty(cls,
                        embedder_name: str,
                        method: ZeroShotMethod,
                        output_dir: Path) -> ZeroShotContactCachedResults:
        report_file_path = output_dir / cls.get_file_name(method)
        if report_file_path.exists():
            report = cls.from_json_file(report_file_path)
            assert report.embedder_name == embedder_name and report.method == method
            return report
        return cls.empty(embedder_name, method)

    def maybe_cached_result(self, seq_id: str) -> Optional[ContactSingleProteinResult]:
        return self.per_protein_results.get(seq_id, None)

    def update_and_sync(self, result: ContactSingleProteinResult, output_dir: Path):
        self.per_protein_results[result.protein_name] = result
        self._write_to_file(output_dir=output_dir)

    def _write_to_file(self, output_dir: Union[Path, str]):
        file_path = output_dir / self.get_file_name(method=self.method)
        with open(file_path, 'w') as f:
            f.write(self.model_dump_json(indent=4))


class AutoEvalReport(BaseModel):
    embedder_name: str = Field(description="Name of the embedder")
    training_date: str = Field(description="Date of training")

    # Results
    supervised_results: Dict[str, SupervisedFrameworkReport] = Field(description="Supervised autoeval results")
    zeroshot_results: Dict[str, ZeroShotFrameworkReport] = Field(description="Zero-Shot autoeval results")
    zeroshot_contact_results: Dict[str, ContactFrameworkReport] = Field(default_factory=dict,
                                                                        description="Zero-Shot contact autoeval results")
    supervised_contact_results: Dict[str, ContactFrameworkReport] = Field(default_factory=dict,
                                                                          description="Supervised contact autoeval results")

    @staticmethod
    def get_file_name(embedder_name):
        return f'autoeval_report_{embedder_name.replace("/", "-")}.json'

    @classmethod
    def empty(cls, embedder_name: str, training_date: str) -> AutoEvalReport:
        return cls(embedder_name=embedder_name, training_date=training_date, supervised_results={}, zeroshot_results={})

    @classmethod
    def loaded_or_empty(cls, embedder_name: str, training_date: str, output_dir: Path) -> AutoEvalReport:
        report_file_path = output_dir / cls.get_file_name(embedder_name)
        if report_file_path.exists():
            report = cls.from_json_file(report_file_path)
            assert report.embedder_name == embedder_name
            return report
        return cls.empty(embedder_name, training_date)

    @classmethod
    def from_json_file(cls, file_path: Union[Path, str]) -> AutoEvalReport:
        """Load AutoEvalReport from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())

    def add_framework_report(self, framework_name: str,
                             framework_mode: AutoEvalMode,
                             report: FrameworkReport):
        match framework_mode:
            case AutoEvalMode.SUPERVISED:
                assert isinstance(report, SupervisedFrameworkReport)
                self.supervised_results[framework_name] = report
            case AutoEvalMode.ZERO_SHOT:
                assert isinstance(report, ZeroShotFrameworkReport)
                self.zeroshot_results[framework_name] = report
            case AutoEvalMode.ZERO_SHOT_CONTACT:
                assert isinstance(report, ContactFrameworkReport)
                self.zeroshot_contact_results[framework_name] = report
            case AutoEvalMode.SUPERVISED_CONTACT_ATTENTION:
                assert isinstance(report, ContactFrameworkReport)
                self.supervised_contact_results[framework_name] = report
            case _:
                raise ValueError(f"Invalid framework mode: {framework_mode}")

    def _all_results(self):
        return [self.supervised_results,
                self.zeroshot_results,
                self.zeroshot_contact_results,
                self.supervised_contact_results]

    def all_framework_names(self):
        return set(framework_name for results in self._all_results() for framework_name in results)

    def maybe_framework_result(self, framework_name: str) -> Optional[FrameworkReport]:
        for results in self._all_results():
            if framework_name in results:
                return results[framework_name]
        return None

    def write(self, output_dir: Path):
        report_name = output_dir / self.get_file_name(self.embedder_name)

        print(f'Writing autoeval report to: {report_name}')
        with open(report_name, 'w') as report_file:
            report_file.write(self.model_dump_json(indent=4))

    def get_uid(self) -> str:
        h = hashlib.sha1()
        h.update(self.embedder_name.encode("utf-8"))
        h.update(self.training_date.encode("utf-8"))
        for result in self._all_results():
            h.update(str(len(result)).encode("utf-8"))
        return h.hexdigest()

    def summary(self, development_mode: bool = False):
        print(f"Autoeval report for {self.embedder_name} on {self.training_date}.")
        for framework_name, report in self.supervised_results.items():
            print(f"\n{framework_name} supervised results:")
            report.summary(development_mode=development_mode)
        for framework_name, report in self.zeroshot_results.items():
            print(f"\n{framework_name} zero-shot results:")
            report.summary(development_mode=development_mode)
        for framework_name, report in self.zeroshot_contact_results.items():
            print(f"\n{framework_name} zero-shot contact results:")
            report.summary()
        for framework_name, report in self.supervised_contact_results.items():
            print(f"\n{framework_name} supervised contact results:")
            report.summary()

    def embedding_stats(self):
        print(f"Embedding stats in autoeval report for {self.embedder_name} on {self.training_date}.")
        for framework_name, report in self.supervised_results.items():
            print(f"\n{framework_name} - embedding stats:")
            print(report.accumulated_embedding_stats())

    def is_development(self) -> bool:
        """ Checks if any of the reports is in development mode """
        all_results = self._all_results()
        all_reports = [r for results in all_results for r in results.values()]
        return any(rep.used_development_mode() for rep in all_reports)

    ## TODO Move to client
    # def compare_with_public_leaderboard(self):
    #    """
    #    Compare this report to the public leaderboard. This implies uploading the report to the autoeval service
    #    temporarily. The report will automatically be deleted after one day.
    #    """
    #    client = AutoEvalServiceClient.default_service()
    #    uid = client.store_comparison_report(report=self.model_dump())
    #    if uid is not None:
    #        print(f"Report stored in the autoeval service with UID: {uid}\n"
    #              f"Open https://autoeval.biocentral.cloud/?uid={uid} to compare.")
#
## TODO Move to client
# def publish(self, name: str, email: str, citation: Optional[str] = None):
#    """
#    Publish this report to the public autoeval dashboard.
#
#    :param name: Name of the publisher
#    :param email: E-Mail of the publisher
#    :param citation: Optional citation for the report. Should have https://doi.org/... format.
#    """
#    client = AutoEvalServiceClient.default_service()
#    client.publish_report(report=self.model_dump(), name=name, email=email, citation=citation)
#
