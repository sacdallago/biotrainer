import requests

from typing import Optional, Dict, Any, List
from biotrainer_core.data_classes.autoeval import AutoEvalReport, AutoEvalPublishedReport


class _AutoEvalServiceClient:
    """ Raw client handling connection to the autoeval service """
    def __init__(self, base_url: str):
        base_url = base_url.rstrip("/")
        self.api_url = f"{base_url}/api/v1/autoeval"

    @classmethod
    def default_service(cls):
        return cls("https://autoeval.biocentral.cloud")

    @staticmethod
    def _handle_response(response, expect_field: Optional[str] = None):
        json = None
        try:
            json = response.json()
        except Exception:
            response.raise_for_status()

        error = json.get("error")
        if error is not None and len(error) > 0:
            raise Exception("[AUTOEVAL-SERVICE] - Error: " + error)
        message = json.get("message")
        if message is not None and len(message) > 0:
            print("[AUTOEVAL-SERVICE] - " + message)
        detail = json.get("detail", [{}])[0].get("msg", None)
        if detail is not None and len(detail) > 0:
            raise Exception("[AUTOEVAL-SERVICE] - Error: " + detail)

        if expect_field is not None:
            if json.get(expect_field) is None:
                raise Exception(f"[AUTOEVAL-SERVICE] - "
                                f"Error: Expected field {expect_field} not found in response: {json}")
        return json

    def get_public_reports(self) -> Dict:
        """Retrieve all published autoeval reports"""
        response = requests.get(self.api_url)
        json = self._handle_response(response, expect_field="reports")
        return json

    def publish_report(self, report: Dict, name: str, email: str, citation: Optional[str] = None) -> None:
        """Publish a new autoeval report
        
        Args:
            report: Converted AutoEvalReport dict
            name: Name of the publisher
            email: Email of the publisher
            citation: Optional citation (must be a valid DOI)
        """
        payload = {
            "report": report,
            "name": name,
            "email": email,
            "citation": citation
        }

        response = requests.post(f"{self.api_url}/publish/", json=payload)
        _ = self._handle_response(response)
        print("You can now view the report at: https://autoeval.biocentral.cloud/")

    def store_comparison_report(self, report: Dict) -> str:
        """Store a report temporarily for comparison
        
        Args:
            report: AutoEvalReport dict
        """

        payload = {"report": report}

        response = requests.post(f"{self.api_url}/compare/", json=payload)
        json = self._handle_response(response, expect_field="uid")
        return json.get("uid")

    def get_comparison_report(self, uid: str) -> Dict[str, Any]:
        """Retrieve a stored report for comparison"""
        response = requests.get(f"{self.api_url}/compare/{uid}")
        json = self._handle_response(response, "report")
        return json.get("report")


class AutoEvalClient:
    """ AutoEvalClient for publishing, comparing and downloading reports """

    def __init__(self, base_url: Optional[str] = None):
        self._client = _AutoEvalServiceClient(base_url) if base_url else _AutoEvalServiceClient.default_service()

    def publish_report(self, report: AutoEvalReport, name: str, email: str, citation: Optional[str] = None):
        """
        Publish this report to the public autoeval dashboard.

        :param report: AutoEval Report to publish
        :param name: Name of the publisher
        :param email: E-Mail of the publisher
        :param citation: Optional citation for the report. Must have https://doi.org/... format.
        """

        report_dict = report.model_dump()
        self._client.publish_report(report=report_dict, name=name, email=email, citation=citation)

    def get_comparison_report(self, uid: str) -> AutoEvalReport:
        """ Retrieve a stored comparison report by uid from the autoeval service. """
        response_json = self._client.get_comparison_report(uid)
        report = AutoEvalReport.model_validate(response_json)
        return report

    def compare_with_public_leaderboard(self, report: AutoEvalReport):
        """
        Compare a report to the public leaderboard. This implies uploading the report to the autoeval service
        temporarily. The report will automatically be deleted after one day.
        """
        uid = self._client.store_comparison_report(report=report.model_dump())
        if uid is not None:
            print(f"Report stored in the autoeval service with UID: {uid}\n"
                  f"Open https://autoeval.biocentral.cloud/?uid={uid} to compare.")

    def get_public_reports(self) -> List[AutoEvalPublishedReport]:
        """Retrieve all published autoeval reports"""
        response_json = self._client.get_public_reports()
        reports = response_json.get("reports", [])
        published_reports = [AutoEvalPublishedReport.model_validate(report) for report in reports]
        return published_reports
