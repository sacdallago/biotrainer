from typing import Dict, Any

from ..core import AutoEvalTask, AutoEvalConfigBank


class PGYMConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        return {}
