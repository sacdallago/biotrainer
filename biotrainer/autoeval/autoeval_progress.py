from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AutoEvalProgress:
    completed_tasks: int
    total_tasks: int
    current_framework_name: str
    current_task_name: str
    final_report: Optional[Dict[str, Any]] = None
