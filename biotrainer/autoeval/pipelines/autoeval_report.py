from __future__ import annotations

from ruamel import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional, List

from ..core import AutoEvalTask
from ..pbc.pbc_datasets import PBC_DATASETS
from ..flip.flip_datasets import FLIP_DATASETS

from ...bioengineer import ZeroShotMethod, RankingResult


class FrameworkReport(ABC):
    @abstractmethod
    def summary(self):
        raise NotImplementedError

    @abstractmethod
    def number_tasks(self):
        raise NotImplementedError

    @abstractmethod
    def get_task_names(self) -> List[str]:
        raise NotImplementedError


class SupervisedFrameworkReport(BaseModel, FrameworkReport):
    min_seq_len: Optional[int] = Field(default=None, description="Minimum sequence length used during evaluation")
    max_seq_len: Optional[int] = Field(default=None, description="Maximum sequence length used during evaluation")
    results: Dict[str, Dict[str, Any]] = Field(description="Supervised autoeval results")

    @classmethod
    def empty(cls, min_seq_len: Optional[int], max_seq_len: Optional[int]) -> SupervisedFrameworkReport:
        return cls(min_seq_len=min_seq_len, max_seq_len=max_seq_len, results={})

    def update_result(self, combined_task_name: str, result: Dict[str, Any]):
        self.results[combined_task_name] = result

    @staticmethod
    def maybe_load_existing_result(embedder_name: str, task_output_dir: Path):
        task_out_file_path = task_output_dir / "out.yml"
        if not task_out_file_path.exists():
            return None
        try:
            with (open(task_out_file_path, 'r') as output_file):
                task_output = yaml.load(output_file, Loader=yaml.RoundTripLoader)
                if task_output["config"]["embedder_name"] == embedder_name and "test_results" in task_output:
                    return task_output
                return None  # File does not seem to be valid
        except Exception:
            return None

    def summary(self):
        print(f"(Minimum sequence length: {self.min_seq_len}, Maximum sequence length: {self.max_seq_len})")
        task_names = self.results.keys()
        print(f"Total tasks: {len(task_names)}")
        print("Results:")

        for task_name in task_names:
            metrics = self.extract_metrics(task_name)
            for metric in metrics:
                print(
                    f"{metric['task_name']} ({metric['protocol']}) - {metric['test_set_name']} - "
                    f"{metric['evaluation_metric']}: {metric['mean']} ({metric['lower']} - {metric['upper']})"
                )

    def extract_metrics(self, combined_task_name: str) -> list[dict]:
        """Extract metrics for a given task."""
        framework_to_datasets = {"PBC": PBC_DATASETS, "FLIP": FLIP_DATASETS}

        framework_name, dataset_name, split_name = AutoEvalTask.split_combined_name(combined_task_name)
        datasets = framework_to_datasets[framework_name.upper()]
        evaluation_metric = datasets[dataset_name]["evaluation_metric"]
        protocol = datasets[dataset_name]["protocol"].name
        test_results = self.results[combined_task_name]["test_results"]

        metrics = []
        for test_set_name, test_set_dict in test_results.items():
            bootstrapping = test_set_dict["bootstrapping"]["results"]
            bootstrapping = {b_dict["name"]: b_dict for b_dict in bootstrapping}
            metric_mean = round(bootstrapping[evaluation_metric]["mean"], 3)
            metric_lower = round(bootstrapping[evaluation_metric]["lower"], 3)
            metric_upper = round(bootstrapping[evaluation_metric]["upper"], 3)

            metrics.append({
                "task_name": combined_task_name,
                "protocol": protocol,
                "test_set_name": test_set_name,
                "evaluation_metric": evaluation_metric,
                "mean": metric_mean,
                "lower": metric_lower,
                "upper": metric_upper
            })
        return metrics

    def number_tasks(self):
        return len(self.results.keys())

    def get_task_names(self) -> List[str]:
        return list(self.results.keys())


class ZeroShotFrameworkReport(BaseModel, FrameworkReport):
    method: ZeroShotMethod = Field(description="Scoring method used")
    aggregated_results: Dict[str, RankingResult] = Field(description="Accumulated autoeval task results "
                                                                     "(combined_task_name -> RankingResult)")
    individual_results: Dict[str, RankingResult] = Field(description="Individual autoeval task results "
                                                                     "(dataset_name -> RankingResult)")

    @classmethod
    def empty(cls, method: ZeroShotMethod) -> ZeroShotFrameworkReport:
        return cls(method=method, aggregated_results={}, individual_results={})

    def aggregate(self, task_name: str, individual_results: Dict[str, RankingResult]):
        self.individual_results.update(individual_results)
        self.aggregated_results[task_name] = RankingResult.aggregate(list(individual_results.values()))

    def summary(self):
        print(f"Zero-shot method: {self.method.value}")
        print(f"Total tasks: {len(self.aggregated_results)}")
        print("Results:")
        for combined_task_name, result in self.aggregated_results.items():
            print(f"{combined_task_name}: "
                  f"\t SCC:  {result.scc_score()}"
                  f"\t NDCG: {result.ndcg_score()}")

    def number_tasks(self):
        return len(self.aggregated_results)

    def get_task_names(self) -> List[str]:
        return list(self.aggregated_results.keys())


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
                        output_dir: Path) -> Optional[ZeroShotCachedResults]:
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


class AutoEvalReport(BaseModel):
    embedder_name: str = Field(description="Name of the embedder")
    training_date: str = Field(description="Date of training")
    supervised_results: Dict[str, SupervisedFrameworkReport] = Field(description="Supervised autoeval results")
    zeroshot_results: Dict[str, ZeroShotFrameworkReport] = Field(description="Zero-Shot autoeval results")

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

    def add_supervised_result(self, framework_name: str, report: SupervisedFrameworkReport):
        self.supervised_results[framework_name] = report

    def add_zeroshot_result(self, framework_name: str, report: ZeroShotFrameworkReport):
        self.zeroshot_results[framework_name] = report

    def maybe_framework_result(self, framework_name: str) -> Optional[FrameworkReport]:
        return self.supervised_results.get(framework_name, self.zeroshot_results.get(framework_name, None))

    def write(self, output_dir: Path):
        report_name = output_dir / self.get_file_name(self.embedder_name)

        print(f'Writing autoeval report to: {report_name}')
        with open(report_name, 'w') as report_file:
            report_file.write(self.model_dump_json(indent=4))

    def summary(self):
        print(f"Autoeval report for {self.embedder_name} on {self.training_date}.")
        for framework_name, report in self.supervised_results.items():
            print(f"\n{framework_name} supervised results:")
            report.summary()
        for framework_name, report in self.zeroshot_results.items():
            print(f"\n{framework_name} zero-shot results:")
            report.summary()

    def compare(self, other_reports: list[AutoEvalReport], plot: Optional[bool] = False):
        """Compare this report with other reports on the same evaluation metrics.
    
        Args:
            other_reports: List of AutoEvalReport objects to compare with
            plot: Whether to plot the comparison
        """
        import pandas as pd
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        all_reports = [self] + other_reports

        # Get all unique task names across all reports
        all_task_names = set()
        for report in all_reports:
            for framework_report in [*report.supervised_results.values(), *report.zeroshot_results.values()]:
                all_task_names.update(framework_report.get_task_names())

        # Create list to store all data
        data = []

        # For each task, gather metrics across all reports
        for task_name in sorted(all_task_names):
            for report in all_reports:
                # TODO Zero-Shot Results
                for framework_report in [*report.supervised_results.values()]:
                    if task_name in framework_report.get_task_names():
                        metrics = framework_report.extract_metrics(task_name)
                        for metric in metrics:
                            data.append({
                                'Task': task_name,
                                'Protocol': metric['protocol'],
                                'Test Set': metric['test_set_name'],
                                'Metric': metric['evaluation_metric'],
                                'Model': report.embedder_name,
                                'Score': f"{metric['mean']} ({metric['lower']}-{metric['upper']})",
                                'Mean': metric['mean'],  # Add numerical values for plotting
                                'Lower': metric['lower'],
                                'Upper': metric['upper']
                            })

        # Create DataFrame and pivot it
        df = pd.DataFrame(data)
        df_pivot = df.pivot_table(
            index=['Task', 'Protocol', 'Test Set', 'Metric'],
            columns='Model',
            values='Score',
            aggfunc='first',
            sort=False
        ).reset_index()

        print("\nComparison of reports:")
        print(df_pivot.to_string(index=False))

        if plot:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt

                # Create combined task+test_set names for plotting
                plot_data = pd.DataFrame(data)
                plot_data['Task_TestSet'] = plot_data.apply(
                    lambda x: f"{x['Task']}\n({x['Test Set']} - {x['Metric']})" if x['Test Set'] != 'test'
                    else f"{x['Task']}\n({x['Metric']})", axis=1
                )

                # Create figure with adjusted size based on number of unique task+test_set combinations
                unique_task_testsets = plot_data['Task_TestSet'].unique()
                plt.figure(figsize=(16, max(8, len(unique_task_testsets) * 0.75)))

                # Set style and create bar plot
                sns.set_style("whitegrid")
                ax = sns.barplot(
                    data=plot_data,
                    x='Task_TestSet',
                    y='Mean',
                    hue='Model',
                    capsize=0.2,
                    err_kws={'linewidth': 1},  # Replace deprecated errwidth
                    errorbar=lambda x: (x.std(), x.std()),  # Symmetric error bars
                    alpha=0.8
                )

                # Add error bars manually using the confidence intervals
                for i, model in enumerate(plot_data['Model'].unique()):
                    model_data = plot_data[plot_data['Model'] == model]
                    for j, task_testset in enumerate(unique_task_testsets):
                        task_data = model_data[model_data['Task_TestSet'] == task_testset]
                        if not task_data.empty:
                            x = j + (i - len(plot_data['Model'].unique()) / 2 + 0.5) * (
                                    0.8 / len(plot_data['Model'].unique()))
                            plt.vlines(
                                x=x,
                                ymin=task_data['Lower'].iloc[0],
                                ymax=task_data['Upper'].iloc[0],
                                color=plt.cm.tab10(i),
                                linewidth=2,
                                alpha=0.6
                            )

                # Customize plot
                plt.title('Performance Comparison Across Tasks')
                plt.xlabel('Task')
                plt.ylabel('Score')

                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')

                # Adjust legend position
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                # Adjust layout to prevent label cutoff
                plt.tight_layout()

                plt.show()

            except ImportError as e:
                print(f"Plotting requires matplotlib and seaborn: {e}")
