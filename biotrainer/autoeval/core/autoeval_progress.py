from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from .report_manager import AutoEvalReport

class AutoEvalProgress(BaseModel):
    completed_tasks: int = Field(description="Number of completed autoeval tasks", ge=0)
    total_tasks: int = Field(description="Total number of autoeval tasks", ge=0)
    current_framework_name: str = Field(description="Name of the current framework that is being evaluated")
    current_task_name: str = Field(description="Name of the current task that is being executed")
    final_report: Optional[AutoEvalReport] = None
