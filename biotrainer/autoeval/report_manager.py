import json

from pathlib import Path
from typing import Dict, Any, Union

from .data_handler import AutoEvalTask


class ReportManager:
    def __init__(self, embedder_name: str, training_date, min_seq_len: int, max_seq_len: int):
        self.embedder_name = embedder_name
        self.training_date = training_date
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.results = {}

    def add_result(self, task: AutoEvalTask, result_dict: Dict[str, Any]):
        self.results[task.name] = result_dict

    def write(self, output_dir: Union[Path, str]) -> Dict[str, Any]:
        result_dict = {
            "embedder_name": self.embedder_name,
            "training_date": self.training_date,
            "min_seq_len": self.min_seq_len,
            "max_seq_len": self.max_seq_len,
            "results": self.results
        }
        report_name = output_dir / f'autoeval_report_{self.embedder_name.replace("/", "-")}.json'

        print(f'Writing autoeval report to: {report_name}')
        with open(report_name, 'w') as report_file:
            report_file.write(json.dumps(result_dict))

        return result_dict
