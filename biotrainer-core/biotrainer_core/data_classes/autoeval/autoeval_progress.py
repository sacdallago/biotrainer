from typing import Optional
from pydantic import BaseModel, Field

from .autoeval_report import FrameworkReport

class AutoEvalProgress(BaseModel):
    completed_tasks: int = Field(description="Number of completed autoeval tasks", ge=0)
    total_tasks: int = Field(description="Total number of autoeval tasks", ge=0)
    current_framework_name: str = Field(description="Name of the current framework that is being evaluated")
    current_task_name: str = Field(description="Name of the current task that is being executed")
    final_report: Optional[FrameworkReport] = Field(default=None, description="Final Framework Report")

    def pipeline_progress_info(self) -> str:
        return (f"AutoEval Progress: {self.completed_tasks}/{self.total_tasks} tasks completed. "
                f"Current Framework: {self.current_framework_name}. "
                f"Current Task: {self.current_task_name}.")