from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Union, Optional

from ..contact import ContactSingleProteinResult
from ..bioengineer_data_classes import ZeroShotMethod, RankingResult


class ZeroShotCachedResults(BaseModel):
    """ Utility class for storing cached results for zero-shot evaluation """
    embedder_name: str = Field(description="Name of the embedder")
    method: ZeroShotMethod = Field(description="Scoring method used")
    individual_results: Dict[str, RankingResult] = Field(description="Individual autoeval task results "
                                                                     "(dataset_name -> RankingResult)")

    @staticmethod
    def get_file_name(method: ZeroShotMethod):
        return f"zero_shot_cached_results_{method.value}.json"

    @classmethod
    def from_json_file(cls, file_path: Union[Path, str]) -> ZeroShotCachedResults:
        """Load ZeroShotCachedResults from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def empty(cls, embedder_name: str, method: ZeroShotMethod) -> ZeroShotCachedResults:
        return cls(embedder_name=embedder_name, method=method, individual_results={})

    @classmethod
    def loaded_or_empty(cls,
                        embedder_name: str,
                        method: ZeroShotMethod,
                        output_dir: Path) -> ZeroShotCachedResults:
        report_file_path = output_dir / cls.get_file_name(method)
        if report_file_path.exists():
            report = cls.from_json_file(report_file_path)
            assert report.embedder_name == embedder_name and report.method == method
            return report
        return cls.empty(embedder_name, method)

    def maybe_cached_result(self, dataset_name: str) -> Optional[RankingResult]:
        return self.individual_results.get(dataset_name, None)

    def update_and_sync(self, dataset_name: str, result: RankingResult, output_dir: Path):
        self.individual_results[dataset_name] = result
        self._write_to_file(output_dir=output_dir)

    def _write_to_file(self, output_dir: Union[Path, str]):
        file_path = output_dir / self.get_file_name(method=self.method)
        with open(file_path, 'w') as f:
            f.write(self.model_dump_json(indent=4))


class ZeroShotContactCachedResults(BaseModel):
    """ Utility class for storing cached results for zero-shot contact evaluation """
    embedder_name: str = Field(description="Name of the embedder")
    method: ZeroShotMethod = Field(
        description="Contact method used")  # Note - only one applicable zeroshot contact method as of now!
    per_protein_results: Dict[str, ContactSingleProteinResult] = Field(
        description="Cached per protein results, stacking"
                    " up to the final dataset result (seq_id -> ContactSingleProteinResult)")

    @staticmethod
    def get_file_name(method: ZeroShotMethod):
        return f"zero_shot_contact_cached_results_{method.value}.json"

    @classmethod
    def from_json_file(cls, file_path: Union[Path, str]) -> ZeroShotContactCachedResults:
        """Load ZeroShotContactCachedResults from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def empty(cls, embedder_name: str, method: ZeroShotMethod) -> ZeroShotContactCachedResults:
        return cls(embedder_name=embedder_name, method=method, per_protein_results={})

    @classmethod
    def loaded_or_empty(cls,
                        embedder_name: str,
                        method: ZeroShotMethod,
                        output_dir: Path) -> ZeroShotContactCachedResults:
        report_file_path = output_dir / cls.get_file_name(method)
        if report_file_path.exists():
            report = cls.from_json_file(report_file_path)
            assert report.embedder_name == embedder_name and report.method == method
            return report
        return cls.empty(embedder_name, method)

    def maybe_cached_result(self, seq_id: str) -> Optional[ContactSingleProteinResult]:
        return self.per_protein_results.get(seq_id, None)

    def update_and_sync(self, result: ContactSingleProteinResult, output_dir: Path):
        self.per_protein_results[result.protein_name] = result
        self._write_to_file(output_dir=output_dir)

    def _write_to_file(self, output_dir: Union[Path, str]):
        file_path = output_dir / self.get_file_name(method=self.method)
        with open(file_path, 'w') as f:
            f.write(self.model_dump_json(indent=4))
