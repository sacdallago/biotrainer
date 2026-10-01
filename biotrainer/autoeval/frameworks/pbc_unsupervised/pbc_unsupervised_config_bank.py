from typing import Dict, Any
from biotrainer_core.data_classes.autoeval import AutoEvalTask

from ...core import AutoEvalConfigBank


class PBCUnsupervisedConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        # No config for EAT
        if task.dataset_name == "cath":
            return {
                    "protocol": "sequence_to_class",
                    "num_nn": 1,
                    "threshold": None,  # Euclidean distance threshold, None means use THE nearest neighbor regardless of distance alternative: 1.1
            }
        assert False, f"Unknown dataset {task.dataset_name}"
