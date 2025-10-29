from typing import Dict, Any

from ..data_handler import AutoEvalTask
from ..config_bank import AutoEvalConfigBank

class PBCConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        # PBC configs are per dataset, so only first part of task name is relevant
        dataset_name = task.name.split("-")[1]
        assert len(dataset_name) > 0, f"PBC dataset name is empty for task: {task}"

        match dataset_name:
            case "scl":
                return {
                    "protocol": "sequence_to_class",
                    "model_choice": "FNN",
                    "optimizer_choice": "adam",
                    "loss_choice": "cross_entropy_loss",
                    "num_epochs": 20,
                    "use_class_weights": False,
                    "learning_rate": 1e-3,
                    "batch_size": 128,
                    "ignore_file_inconsistencies": True,
                }
            case "secondary_structure":
                return {
                    "protocol": "residue_to_class",
                    "model_choice": "CNN",
                    "optimizer_choice": "adam",
                    "loss_choice": "cross_entropy_loss",
                    "num_epochs": 20,
                    "use_class_weights": False,
                    "learning_rate": 1e-3,
                    "batch_size": 128,
                    "ignore_file_inconsistencies": True,
                }
        raise ValueError(f"Unknown task in config bank: {task}")