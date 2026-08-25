from pydantic import Field
from biotrainer_core.data_classes.autoeval import AutoEvalReport, AutoEvalPublishedReport

class DashboardReport(AutoEvalPublishedReport):
    is_visible: bool = Field(default=True, description="If true, the report is visible in the dashboard")

    @classmethod
    def from_published_report(cls, published_report: AutoEvalPublishedReport):
        return cls(**published_report.model_dump(), is_visible=True)

    @classmethod
    def from_loaded_report(cls, loaded_report: AutoEvalReport):
        return DashboardReport(report=loaded_report, is_visible=True,
                               name="user", email="user@example.com", citation=None, official=False)