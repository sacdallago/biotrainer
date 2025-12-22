from typing import Dict, Any

from ..data_handler import AutoEvalTask
from ..config_bank import AutoEvalConfigBank

class PBCConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        # PBC configs are per dataset, so only first part of task name is relevant
        dataset_name = task.dataset_name
        assert len(dataset_name) > 0, f"PBC dataset name is empty for task: {task}"

        # Common configuration options shared across most tasks
        base_config = {
            "model_choice": "LogReg",
            "num_epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "ignore_file_inconsistencies": True,
        }

        # Task-specific configurations that override or extend base config
        task_specific_configs = {
            "binding": {
                "protocol": "residue_to_class",
                "use_class_weights": True,
            },
            "conservation": {
                "protocol": "residue_to_class",
                "use_class_weights": True,
            },
            "disorder": {
                "protocol": "residue_to_value",
            },
            "membrane": {
                "protocol": "residue_to_class",
                "use_class_weights": True,
            },
            "scl": {
                "protocol": "sequence_to_class",
                "use_class_weights": True,
            },
            "secondary_structure": {
                "protocol": "residue_to_class",
                "use_class_weights": True,
            },
        }

        if dataset_name not in task_specific_configs:
            raise ValueError(f"Unknown task in config bank: {task}")

        config = base_config.copy()
        config.update(task_specific_configs[dataset_name])
    
        return config