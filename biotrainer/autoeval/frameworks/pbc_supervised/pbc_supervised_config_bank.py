from typing import Dict, Any
from biotrainer_core.data_classes.autoeval import AutoEvalTask

from ...core import AutoEvalConfigBank


class PBCConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        # PBC configs are per dataset, so only first part of task name is relevant
        dataset_name = task.dataset_name
        assert len(dataset_name) > 0, f"PBC dataset name is empty for task: {task}"

        # Common configuration options shared across most tasks
        base_config = {
            "num_epochs": 10,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "ignore_file_inconsistencies": True,
            "seed": 42,
            "dropout_rate": 0.0,
            "shuffle": True,
            "patience": 5,
            "epsilon": 1e-3,
            "external_writer": "none",
        }

        # Task-specific configurations that override or extend base config
        task_specific_configs = {
            "conservation": {
                "model_choice": "CNN",
                "protocol": "residue_to_class",
                "use_class_weights": False,
            },
            "disorder_chezod": {
                "model_choice": "CNN",
                "protocol": "residue_to_value",
                "loss_choice": "smooth_l1_loss",
            },
            "disorder_trizod": {
                "model_choice": "CNN",
                "protocol": "residue_to_value",
                "loss_choice": "smooth_l1_loss",
            },
            "frustration-classification": {
                "model_choice": "CNN",
                "protocol": "residue_to_class",
                "use_class_weights": False,
            },
            "frustration-regression": {
                "model_choice": "CNN",
                "protocol": "residue_to_value",
            },
            "membrane": {
                "model_choice": "CNN",
                "protocol": "residue_to_class",
                "use_class_weights": False,
            },
            "phages": {
                "model_choice": "FNN",
                "protocol": "sequence_to_class",
                "use_class_weights": False,
            },
            "scl": {
                "model_choice": "FNN",
                "protocol": "sequence_to_class",
                "use_class_weights": False,
            },
            "secondary_structure": {
                "model_choice": "CNN",
                "protocol": "residue_to_class",
                "use_class_weights": False,
            },
        }

        if dataset_name not in task_specific_configs:
            raise ValueError(f"Unknown task in config bank: {task}")

        config = base_config.copy()
        config.update(task_specific_configs[dataset_name])

        return config
