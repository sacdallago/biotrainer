from pydantic import Field
from biotrainer_core.data_classes.autoeval import AutoEvalReport, AutoEvalPublishedReport

class DashboardReport(AutoEvalPublishedReport):
    is_visible: bool = Field(default=True, description="If true, the report is visible in the dashboard")
    is_loaded: bool = Field(default=False, description="If true, the report is loaded locally")

    @classmethod
    def from_published_report(cls, published_report: AutoEvalPublishedReport):
        return cls(**published_report.model_dump(), is_visible=True)

    @classmethod
    def from_loaded_report(cls, loaded_report: AutoEvalReport):
        return DashboardReport(report=loaded_report, is_visible=True, is_loaded=True,
                               name="user", email="user@example.com", citation=None, official=False)

    def tooltip(self) -> str:
        if self.is_loaded:
            return f"Uploaded locally - can be published to the leaderboards."
        return (f"Published by: {self.name}\n\n"
                f"E-Mail: {self.email}\n\n"
                f"Citation: [{self.citation}]({self.citation})\n\n"
                f"Official: {self.official}")