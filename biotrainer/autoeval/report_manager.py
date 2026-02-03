from __future__ import annotations

from ruamel import yaml
from pathlib import Path
from typing import Dict, Any, Union
from pydantic import BaseModel, Field

from .interfaces import AutoEvalTask
from .pbc.pbc_datasets import PBC_DATASETS
from .flip.flip_datasets import FLIP_DATASETS


class AutoEvalReport(BaseModel):
    embedder_name: str = Field(description="Name of the embedder")
    training_date: str = Field(description="Date of training")
    min_seq_len: int = Field(description="Minimum sequence length used during evaluation")
    max_seq_len: int = Field(description="Maximum sequence length used during evaluation")
    results: Dict[str, Dict[str, Any]] = Field(description="Autoeval results")

    @classmethod
    def from_json_file(cls, file_path: Union[Path, str]) -> AutoEvalReport:
        """Load AutoEvalReport from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())

    def _extract_metrics(self, task_name: str) -> list[dict]:
        """Extract metrics for a given task."""
        framework_to_datasets = {"PBC": PBC_DATASETS, "FLIP": FLIP_DATASETS}

        framework_name, dataset_name, split_name = AutoEvalTask.split_combined_name(task_name)
        datasets = framework_to_datasets[framework_name.upper()]
        evaluation_metric = datasets[dataset_name]["evaluation_metric"]
        protocol = datasets[dataset_name]["protocol"].name
        test_results = self.results[task_name]["test_results"]

        metrics = []
        for test_set_name, test_set_dict in test_results.items():
            bootstrapping = test_set_dict["bootstrapping"]["results"]
            metric_mean = round(bootstrapping[evaluation_metric]["mean"], 3)
            metric_lower = round(bootstrapping[evaluation_metric]["lower"], 3)
            metric_upper = round(bootstrapping[evaluation_metric]["upper"], 3)

            metrics.append({
                "task_name": task_name,
                "protocol": protocol,
                "test_set_name": test_set_name,
                "evaluation_metric": evaluation_metric,
                "mean": metric_mean,
                "lower": metric_lower,
                "upper": metric_upper
            })
        return metrics

    def summary(self):
        print(f"Autoeval report for {self.embedder_name} on {self.training_date}.")
        print(f"(Minimum sequence length: {self.min_seq_len}, Maximum sequence length: {self.max_seq_len})")
        task_names = self.results.keys()
        print(f"Total tasks: {len(task_names)}")
        print("Results:")

        for task_name in task_names:
            metrics = self._extract_metrics(task_name)
            for metric in metrics:
                print(
                    f"{metric['task_name']} ({metric['protocol']}) - {metric['test_set_name']} - "
                    f"{metric['evaluation_metric']}: {metric['mean']} ({metric['lower']} - {metric['upper']})"
                )

    def compare(self, other_reports: list[AutoEvalReport]):
        """Compare this report with other reports on the same evaluation metrics.

        Args:
            other_reports: List of AutoEvalReport objects to compare with
        """
        import pandas as pd
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        all_reports = [self] + other_reports
        report_names = [report.embedder_name for report in all_reports]

        # Get all unique task names across all reports
        all_task_names = set()
        for report in all_reports:
            all_task_names.update(report.results.keys())

        # Create list to store all data
        data = []

        # For each task, gather metrics across all reports
        for task_name in sorted(all_task_names):
            for report in all_reports:
                if task_name in report.results:
                    metrics = report._extract_metrics(task_name)
                    for metric in metrics:
                        data.append({
                            'Task': task_name,
                            'Protocol': metric['protocol'],
                            'Test Set': metric['test_set_name'],
                            'Metric': metric['evaluation_metric'],
                            'Model': report.embedder_name,
                            'Score': f"{metric['mean']} ({metric['lower']}-{metric['upper']})"
                        })

        # Create DataFrame and pivot it
        df = pd.DataFrame(data)
        df_pivot = df.pivot_table(
            index=['Task', 'Protocol', 'Test Set', 'Metric'],
            columns='Model',
            values='Score',
            aggfunc='first'
        ).reset_index()

        print("\nComparison of reports:")
        print(df_pivot.to_string(index=False))


class ReportManager:
    def __init__(self, embedder_name: str, training_date, min_seq_len: int, max_seq_len: int):
        self.embedder_name = embedder_name
        self.training_date = training_date
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.results = {}

    def maybe_load_existing_result(self, task_output_dir: Path):
        task_out_file_path = task_output_dir / "out.yml"
        if not task_out_file_path.exists():
            return None
        try:
            with (open(task_out_file_path, 'r') as output_file):
                task_output = yaml.load(output_file, Loader=yaml.RoundTripLoader)
                if task_output["config"]["embedder_name"] == self.embedder_name and "test_results" in task_output:
                    return task_output
                return None  # File does not seem to be valid
        except Exception:
            return None

    def add_result(self, task: AutoEvalTask, result_dict: Dict[str, Any]):
        self.results[task.combined_name()] = result_dict

    def write(self, output_dir: Union[Path, str]) -> AutoEvalReport:
        report = AutoEvalReport(embedder_name=self.embedder_name,
                                training_date=self.training_date,
                                min_seq_len=self.min_seq_len,
                                max_seq_len=self.max_seq_len,
                                results=self.results)

        report_name = output_dir / f'autoeval_report_{self.embedder_name.replace("/", "-")}.json'

        print(f'Writing autoeval report to: {report_name}')
        with open(report_name, 'w') as report_file:
            report_file.write(report.model_dump_json(indent=4))

        return report
