from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AutoEvalProgress:
    current_task: int
    total_tasks: int
    final_report: Optional[Dict[str, Any]] = None
