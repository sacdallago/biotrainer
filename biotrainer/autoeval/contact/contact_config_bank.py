from typing import Dict, Any

from ..core import AutoEvalTask, AutoEvalConfigBank

class ContactConfigBank(AutoEvalConfigBank): # can be common for zeroshot and supervised contact tasks!

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        return {}