from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, Any, Union
from pydantic import BaseModel, Field

from .data_handler import AutoEvalTask
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

    def summary(self):
        framework_to_datasets = {"PBC": PBC_DATASETS, "FLIP": FLIP_DATASETS}

        print(f"Autoeval report for {self.embedder_name} on {self.training_date}.")
        print(f"(Minimum sequence length: {self.min_seq_len}, Maximum sequence length: {self.max_seq_len})")
        task_names = self.results.keys()
        print(f"Total tasks: {len(task_names)}")
        print("Results:")
        for task_name in task_names:
            framework_name, dataset_name, split_name = AutoEvalTask.split_combined_name(task_name)
            datasets = framework_to_datasets[framework_name.upper()]
            evaluation_metric = datasets[dataset_name]["evaluation_metric"]
            protocol = datasets[dataset_name]["protocol"].name
            test_results = self.results[task_name]["test_results"]
            for test_set_name, test_set_dict in test_results.items():
                bootstrapping = test_set_dict["bootstrapping"]["results"]
                metric_mean = round(bootstrapping[evaluation_metric]["mean"], 3)
                metric_lower = round(bootstrapping[evaluation_metric]["lower"], 3)
                metric_upper = round(bootstrapping[evaluation_metric]["upper"], 3)
                print(
                    f"{task_name} ({protocol}) - {test_set_name} - {evaluation_metric}: "
                    f"{metric_mean} ({metric_lower} - {metric_upper})"
                )


class ReportManager:
    def __init__(self, embedder_name: str, training_date, min_seq_len: int, max_seq_len: int):
        self.embedder_name = embedder_name
        self.training_date = training_date
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.results = {}

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
            report_file.write(json.dumps(report.model_dump_json(indent=4)))

        return report
