from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


class AutoEvalTask(BaseModel):
    framework_name: str = Field(description="Name of the framework of this task")
    dataset_name: str = Field(description="Name of the dataset of this task")
    split_name: Optional[str] = Field(default=None, description="Name of the split of this task (optional)")
    input_files: List[Path] = Field(description="Path(s) to the input file(s) of this task")
    type: str = Field(description="Type of the task (e.g. protein/dna)")

    def combined_name(self):
        return f"{self.framework_name}-{self.dataset_name}-{self.split_name}" if self.split_name else \
            f"{self.framework_name}-{self.dataset_name}"

    @staticmethod
    def split_combined_name(combined_name: str) -> tuple[str, str, Optional[str]]:
        vals = combined_name.split("-")
        framework_name, dataset_name, split_name = vals[0], vals[1], vals[2] if len(vals) > 2 else None
        return framework_name, dataset_name, split_name
