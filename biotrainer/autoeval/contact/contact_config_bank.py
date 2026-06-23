from typing import Dict, Any

from ..core import AutoEvalTask, AutoEvalConfigBank

# NOTE: this is to be extended later to include supervised contact tasks!
class ContactConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        return {}