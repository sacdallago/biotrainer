from typing import Optional
from pydantic import BaseModel, Field, field_validator, EmailStr

from .autoeval_report import AutoEvalReport


class AutoEvalPublishedReport(BaseModel):
    """ Extension of the AutoEvalReport class to include additional information for
    publishing reports to the AutoEval service. """
    report: AutoEvalReport = Field(description="Report to publish")
    name: str = Field(description="Name of the publisher", min_length=3)
    email: EmailStr = Field(description="Email of the publisher", min_length=5)
    official: bool = Field(default=False, description="Whether the report is part of the official leaderboard or not")
    citation: Optional[str] = Field(default=None, description="Citation to be used for the model, must be a valid DOI")

    @field_validator('citation')
    def validate_citation(cls, v):
        if v is not None and not str(v).lower().startswith("https://doi.org/"):
            raise ValueError("Citation must be a valid DOI URL")
        if v is not None and len(v) > 150:
            raise ValueError("Citation must be less than 150 characters")
        return v